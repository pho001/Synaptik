package Backend.kernels.cpu;

import Config.backend.CpuKernelConfig;
import Config.backend.SumAccuracyMode;
import Config.backend.VectorPolicy;
import Operations.FusedOperation;
import Operations.Operation;
import Tensor.Tensor;

public final class  CpuExecutionConfig {
    public static final class ResolvedDispatchHints {
        private final int totalLength;
        private final CpuExecutionMode mode;
        private final int scalarChunkSize;
        private final int vectorChunkSize;

        public ResolvedDispatchHints(int totalLength, CpuExecutionMode mode, int scalarChunkSize, int vectorChunkSize) {
            this.totalLength = Math.max(0, totalLength);
            this.mode = mode == null ? CpuExecutionMode.SCALAR : mode;
            this.scalarChunkSize = Math.max(1, scalarChunkSize);
            this.vectorChunkSize = Math.max(1, vectorChunkSize);
        }

        private boolean matchesLength(int length) {
            return totalLength == Math.max(0, length);
        }
    }

    private static final ThreadLocal<ResolvedDispatchHints> RESOLVED_HINTS = new ThreadLocal<>();

    private final int matMulTileM;
    private final int matMulTileN;
    private final int matMulTileK;
    private final int vectorMinSize;
    private final int parallelMinSize;
    private final int matMulParallelMinSize;
    private final int parallelism;
    private final int chunksPerWorker;
    private final int minChunkSize;
    private final int contiguousMaterializeThreshold;
    private final SumAccuracyMode sumAccuracyMode;
    private final double lowCostNsPerElementThreshold;
    private final VectorPolicy vectorPolicyCheap;
    private final VectorPolicy vectorPolicyTranscendental;
    private final VectorPolicy vectorPolicyReduction;

    public CpuExecutionConfig(
            int vectorMinSize,
            int matMulTileM,
            int matMulTileN,
            int matMulTileK,
            int parallelMinSize,
            int matMulParallelMinSize,
            int parallelism,
            int chunksPerWorker,
            int minChunkSize,
            int contiguousMaterializeThreshold,
            SumAccuracyMode sumAccuracyMode,
            double lowCostNsPerElementThreshold,
            VectorPolicy vectorPolicyCheap,
            VectorPolicy vectorPolicyTranscendental,
            VectorPolicy vectorPolicyReduction
    ) {
        this.vectorMinSize = vectorMinSize;
        this.matMulTileM = matMulTileM;
        this.matMulTileN = matMulTileN;
        this.matMulTileK = matMulTileK;
        this.parallelMinSize = parallelMinSize;
        this.matMulParallelMinSize = matMulParallelMinSize;
        this.parallelism = parallelism;
        this.chunksPerWorker = chunksPerWorker;
        this.minChunkSize = minChunkSize;
        this.contiguousMaterializeThreshold = contiguousMaterializeThreshold;
        this.sumAccuracyMode = sumAccuracyMode == null ? SumAccuracyMode.FAST : sumAccuracyMode;
        this.lowCostNsPerElementThreshold = lowCostNsPerElementThreshold <= 0.0d ? 2.0d : lowCostNsPerElementThreshold;
        this.vectorPolicyCheap = vectorPolicyCheap == null ? VectorPolicy.AUTO : vectorPolicyCheap;
        this.vectorPolicyTranscendental = vectorPolicyTranscendental == null ? VectorPolicy.AUTO : vectorPolicyTranscendental;
        this.vectorPolicyReduction = vectorPolicyReduction == null ? VectorPolicy.AUTO : vectorPolicyReduction;
    }

    public static CpuExecutionConfig defaults() {
        return fromKernelConfig(CpuKernelConfig.defaultsTraining());
    }

    public static CpuExecutionConfig fromKernelConfig(CpuKernelConfig config) {
        if (config == null) {
            throw new IllegalArgumentException("config cannot be null");
        }
        SumAccuracyMode sumAccuracyMode = resolveSumAccuracyOverride(config.sumAccuracyMode());
        return new CpuExecutionConfig(
                config.vectorMinSize(),
                config.matMulTileM(),
                config.matMulTileN(),
                config.matMulTileK(),
                config.parallelMinSize(),
                config.matMulParallelMinSize(),
                config.parallelism(),
                config.chunksPerWorker(),
                config.minChunkSize(),
                config.contiguousMaterializeThreshold(),
                sumAccuracyMode,
                config.lowCostNsPerElementThreshold(),
                config.vectorPolicyCheap(),
                config.vectorPolicyTranscendental(),
                config.vectorPolicyReduction()
        );
    }

    private static SumAccuracyMode resolveSumAccuracyOverride(SumAccuracyMode fallback) {
        String raw = System.getProperty("cg.cpu.sumAccuracyOverride");
        if (raw == null || raw.isBlank()) {
            return fallback;
        }
        try {
            return SumAccuracyMode.valueOf(raw.trim().toUpperCase());
        } catch (IllegalArgumentException ignored) {
            return fallback;
        }
    }

    public CpuExecutionMode modeFor(Operation op, Tensor node) {
        ResolvedDispatchHints hints = RESOLVED_HINTS.get();
        if (node != null && hints != null && hints.matchesLength(node.getFlatDataSize())) {
            return hints.mode;
        }
        if (op == null || node == null || !op.isElementWise()) {
            return CpuExecutionMode.SCALAR;
        }
        int size = node.getFlatDataSize();
        VectorPolicy vectorPolicy = resolveElementWisePolicy(op);
        boolean vectorAllowed = isVectorAllowed(vectorPolicy, size);
        if (size >= parallelMinSize) {
            if (vectorAllowed) {
                return CpuExecutionMode.PARALLEL_VECTOR;
            }
            return CpuExecutionMode.PARALLEL;
        }
        if (vectorAllowed) {
            return CpuExecutionMode.VECTOR;
        }
        return CpuExecutionMode.SCALAR;
    }

    public int plannedWorkers() {
        int configured = parallelism > 0 ? parallelism : Runtime.getRuntime().availableProcessors();
        return Math.max(1, configured);
    }

    public int computeChunkSize(int totalLength, int vectorWidth) {
        ResolvedDispatchHints hints = RESOLVED_HINTS.get();
        if (hints != null && hints.matchesLength(totalLength)) {
            if (vectorWidth > 1) {
                return hints.vectorChunkSize;
            }
            return hints.scalarChunkSize;
        }
        int length = Math.max(1, totalLength);
        int workers = plannedWorkers();
        int targets = Math.max(workers, workers * Math.max(1, chunksPerWorker));
        int candidate = (length + targets - 1) / targets;
        int chunk = Math.max(Math.max(1, minChunkSize), candidate);
        if (vectorWidth > 1) {
            int rem = chunk % vectorWidth;
            if (rem != 0) {
                chunk += (vectorWidth - rem);
            }
        }
        return chunk;
    }

    public int contiguousMaterializeThreshold() {
        return contiguousMaterializeThreshold;
    }

    public int matMulTileM() {
        return matMulTileM;
    }

    public int matMulTileN() {
        return matMulTileN;
    }

    public int matMulTileK() {
        return matMulTileK;
    }

    public int parallelMinSize() {
        return parallelMinSize;
    }

    public int matMulParallelMinSize() {
        return matMulParallelMinSize;
    }

    public SumAccuracyMode sumAccuracyMode() {
        return sumAccuracyMode;
    }

    public double lowCostNsPerElementThreshold() {
        return lowCostNsPerElementThreshold;
    }

    public CpuExecutionMode modeForReduction(int workSize) {
        int size = Math.max(1, workSize);
        boolean vectorAllowed = isVectorAllowed(vectorPolicyReduction, size);
        if (size >= parallelMinSize) {
            if (vectorAllowed) {
                return CpuExecutionMode.PARALLEL_VECTOR;
            }
            return CpuExecutionMode.PARALLEL;
        }
        if (vectorAllowed) {
            return CpuExecutionMode.VECTOR;
        }
        return CpuExecutionMode.SCALAR;
    }

    private boolean isVectorAllowed(VectorPolicy policy, int size) {
        return switch (policy) {
            case FORCE_ON -> true;
            case FORCE_OFF -> false;
            case AUTO -> size >= vectorMinSize;
        };
    }

    private VectorPolicy resolveElementWisePolicy(Operation op) {
        Operation.OpType type = op.opType();
        if (type == null) {
            return vectorPolicyCheap;
        }
        switch (type) {
            case LOG, EXP, FAST_EXP, TANH, FAST_TANH, POW, SQRT, SIGMOID -> {
                return vectorPolicyTranscendental;
            }
            case FUSED -> {
                if (op instanceof FusedOperation fused) {
                    return fused.isLowCostHint() ? vectorPolicyCheap : vectorPolicyTranscendental;
                }
                return vectorPolicyTranscendental;
            }
            default -> {
                return vectorPolicyCheap;
            }
        }
    }

    public static void pushResolvedHints(ResolvedDispatchHints hints) {
        if (hints == null) {
            RESOLVED_HINTS.remove();
            return;
        }
        RESOLVED_HINTS.set(hints);
    }

    public static void clearResolvedHints() {
        RESOLVED_HINTS.remove();
    }

}
