package Backend.kernels.cpu;

import Config.backend.CpuKernelConfig;
import Config.backend.SumAccuracyMode;
import Config.backend.VectorPolicy;
import Operations.FusedOperation;
import Operations.Operation;
import Tensor.Tensor;

public final class  CpuExecutionConfig {
    private final int vectorMinSize;
    private final int parallelMinSize;
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
            int parallelMinSize,
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
        this.parallelMinSize = parallelMinSize;
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
        return new CpuExecutionConfig(
                config.vectorMinSize(),
                config.parallelMinSize(),
                config.parallelism(),
                config.chunksPerWorker(),
                config.minChunkSize(),
                config.contiguousMaterializeThreshold(),
                config.sumAccuracyMode(),
                config.lowCostNsPerElementThreshold(),
                config.vectorPolicyCheap(),
                config.vectorPolicyTranscendental(),
                config.vectorPolicyReduction()
        );
    }

    public CpuExecutionMode modeFor(Operation op, Tensor node) {
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

}
