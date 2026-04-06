package backend.kernels.cpu;

import backend.blas.BlasProvider;
import backend.runtime.BlasConfig;
import config.backend.AttentionMatMulPolicy;
import config.backend.CpuKernelConfig;
import config.backend.SumAccuracyMode;
import config.backend.VectorPolicy;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import operations.FusedOperation;
import operations.Operation;
import tensor.DataType;
import tensor.Tensor;

import java.util.Objects;

public final class CpuExecutionPlanner {
    public static final int DEFAULT_MATMUL_TILE_M = 32;
    public static final int DEFAULT_MATMUL_TILE_N = 64;
    public static final int DEFAULT_MATMUL_TILE_K = 64;



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
    private final AttentionMatMulPolicy attentionMatMulPolicy;

    public CpuExecutionPlanner(
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
            VectorPolicy vectorPolicyReduction,
            AttentionMatMulPolicy attentionMatMulPolicy
    ) {
        this.vectorMinSize = Math.max(1, vectorMinSize);
        this.matMulTileM = positiveOrDefault(matMulTileM, DEFAULT_MATMUL_TILE_M);
        this.matMulTileN = positiveOrDefault(matMulTileN, DEFAULT_MATMUL_TILE_N);
        this.matMulTileK = positiveOrDefault(matMulTileK, DEFAULT_MATMUL_TILE_K);
        this.parallelMinSize = Math.max(1, parallelMinSize);
        this.matMulParallelMinSize = Math.max(1, matMulParallelMinSize);
        this.parallelism = Math.max(0, parallelism);
        this.chunksPerWorker = Math.max(1, chunksPerWorker);
        this.minChunkSize = Math.max(1, minChunkSize);
        this.contiguousMaterializeThreshold = Math.max(0, contiguousMaterializeThreshold);
        this.sumAccuracyMode = Objects.requireNonNullElse(sumAccuracyMode, SumAccuracyMode.FAST);
        this.lowCostNsPerElementThreshold = lowCostNsPerElementThreshold > 0.0d ? lowCostNsPerElementThreshold : 2.0d;
        this.vectorPolicyCheap = Objects.requireNonNullElse(vectorPolicyCheap, VectorPolicy.AUTO);
        this.vectorPolicyTranscendental = Objects.requireNonNullElse(vectorPolicyTranscendental, VectorPolicy.AUTO);
        this.vectorPolicyReduction = Objects.requireNonNullElse(vectorPolicyReduction, VectorPolicy.AUTO);
        this.attentionMatMulPolicy = Objects.requireNonNullElse(attentionMatMulPolicy, AttentionMatMulPolicy.AUTO);
    }

    public static CpuExecutionPlanner from(CpuKernelConfig config) {
        Objects.requireNonNull(config, "config cannot be null");
        return new CpuExecutionPlanner(
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
                config.sumAccuracyMode(),
                config.lowCostNsPerElementThreshold(),
                config.vectorPolicyCheap(),
                config.vectorPolicyTranscendental(),
                config.vectorPolicyReduction(),
                config.attentionMatMulPolicy()
        );
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

    public int matMulTileM() {
        return matMulTileM;
    }

    public int matMulTileN() {
        return matMulTileN;
    }

    public int matMulTileK() {
        return matMulTileK;
    }

    public int matMulParallelMinSize() {
        return matMulParallelMinSize;
    }

    public int plannedWorkers() {
        int configured = parallelism > 0 ? parallelism : Runtime.getRuntime().availableProcessors();
        return Math.max(1, configured);
    }

    public boolean shouldMaterializeNonContiguous(int logicalSize) {
        return logicalSize >= contiguousMaterializeThreshold;
    }

    public int preferredVectorWidth(DataType dataType) {
        if (dataType == null) {
            return 1;
        }
        return switch (dataType) {
            case FLOAT64 -> DoubleVector.SPECIES_PREFERRED.length();
            case FLOAT32 -> FloatVector.SPECIES_PREFERRED.length();
            case BFLOAT16 -> 1;
            case INT32 -> 1;
            case BOOL -> 1;
        };
    }

    public ResolvedDispatchHints resolveDispatchHints(Operation op, Tensor node) {
        if (op == null || node == null
                || (op.opType().category() != Operation.OpArityClass.ELEMENT_WISE && op.opType() != Operation.OpType.FUSED)) {
            return new ResolvedDispatchHints(0, CpuExecutionMode.SCALAR, 1, 1, 1, 1);
        }

        int totalLength = Math.max(0, node.getFlatDataSize());
        int vectorWidth = preferredVectorWidth(node.getDataType());
        VectorPolicy policy = resolveElementWisePolicy(op);
        boolean vectorAllowed = vectorWidth > 1 && isVectorAllowed(policy, totalLength, effectiveVectorMinSize(op));

        CpuExecutionMode mode;
        if (totalLength >= effectiveParallelMinSize(op)) {
            mode = vectorAllowed ? CpuExecutionMode.PARALLEL_VECTOR : CpuExecutionMode.PARALLEL;
        } else {
            mode = vectorAllowed ? CpuExecutionMode.VECTOR : CpuExecutionMode.SCALAR;
        }

        return new ResolvedDispatchHints(
                totalLength,
                mode,
                computeChunkSize(totalLength, 1),
                computeChunkSize(totalLength, vectorWidth),
                vectorWidth,
                plannedWorkers()
        );
    }

    public ResolvedReductionHints resolveReductionHints(int logicalSize, DataType dataType) {
        int size = Math.max(0, logicalSize);
        int vectorWidth = preferredVectorWidth(dataType);
        boolean vectorAllowed = vectorWidth > 1 && isVectorAllowed(vectorPolicyReduction, size, vectorMinSize);

        CpuExecutionMode mode;
        if (size >= parallelMinSize) {
            mode = vectorAllowed ? CpuExecutionMode.PARALLEL_VECTOR : CpuExecutionMode.PARALLEL;
        } else {
            mode = vectorAllowed ? CpuExecutionMode.VECTOR : CpuExecutionMode.SCALAR;
        }

        int chunkSize = computeChunkSize(
                size,
                mode == CpuExecutionMode.VECTOR || mode == CpuExecutionMode.PARALLEL_VECTOR ? vectorWidth : 1
        );

        return new ResolvedReductionHints(
                size,
                mode,
                chunkSize,
                vectorWidth,
                plannedWorkers(),
                sumAccuracyMode
        );
    }

    public ResolvedMatMulHints resolveMatMulHints(Tensor a, Tensor b, Tensor out, BlasConfig blasConfig) {
        Objects.requireNonNull(a, "a cannot be null");
        Objects.requireNonNull(b, "b cannot be null");
        Objects.requireNonNull(out, "out cannot be null");
        Objects.requireNonNull(blasConfig, "blasConfig cannot be null");

        int[] as = a.getShapeUnsafe();
        int[] bs = b.getShapeUnsafe();
        if (as.length < 2 || bs.length < 2) {
            throw new IllegalArgumentException("MatMul expects rank >= 2 tensors.");
        }

        int m = as[as.length - 2];
        int k = as[as.length - 1];
        int n = bs[bs.length - 1];
        long batchCount = 1L;
        int outRank = out.getShapeUnsafe().length;
        for (int i = 0; i < outRank - 2; i++) {
            batchCount *= out.getShapeUnsafe()[i];
        }
        long work = batchCount * m * n * k;

        boolean parallel = work >= matMulParallelMinSize && plannedWorkers() > 1;
        boolean useBlas = as.length == 2 && bs.length == 2 && shouldUseBlas(a, b, out, m, n, k, blasConfig);
        boolean useBatchedBlas = as.length > 2 && shouldUseBatchedBlas(a, b, out, m, n, k, work, blasConfig);

        return new ResolvedMatMulHints(
                useBlas,
                useBatchedBlas,
                parallel,
                matMulTileM,
                matMulTileN,
                matMulTileK,
                plannedWorkers(),
                work
        );
    }

    public int computeChunkSize(int totalLength, int alignment) {
        int length = Math.max(1, totalLength);
        int workers = plannedWorkers();
        int targets = Math.max(workers, workers * chunksPerWorker);
        int candidate = (length + targets - 1) / targets;
        int chunk = Math.max(minChunkSize, candidate);

        int align = Math.max(1, alignment);
        if (align > 1) {
            int rem = chunk % align;
            if (rem != 0) {
                chunk += (align - rem);
            }
        }
        return chunk;
    }

    private boolean shouldUseBlas(
            Tensor a,
            Tensor b,
            Tensor out,
            int m,
            int n,
            int k,
            BlasConfig blasConfig
    ) {
        if (out.getDataType() != DataType.FLOAT32 && out.getDataType() != DataType.FLOAT64 && out.getDataType() != DataType.BFLOAT16) {
            return false;
        }
        if (blasConfig.provider() != BlasProvider.OPENBLAS_FFM) {
            return false;
        }
        long work = (long) m * n * k;
        if (work < blasConfig.matMulMinWork()) {
            return false;
        }
        if (!a.isContiguous() || !b.isContiguous() || !out.isContiguous()) {
            return false;
        }
        if (out.getDataType() == DataType.FLOAT32 || out.getDataType() == DataType.BFLOAT16) {
            if (blasConfig.f32RequireMgeK() && m < k) {
                return false;
            }
            if (((double) n / Math.max(1, k)) > blasConfig.f32MaxNOverK()) {
                return false;
            }
        }
        return true;
    }

    private boolean shouldUseBatchedBlas(
            Tensor a,
            Tensor b,
            Tensor out,
            int m,
            int n,
            int k,
            long work,
            BlasConfig blasConfig
    ) {
        if (out.getDataType() != DataType.FLOAT32 && out.getDataType() != DataType.FLOAT64 && out.getDataType() != DataType.BFLOAT16) {
            return false;
        }
        if (!isAttentionLikeBatchedMatMul(a, b, out)) {
            return false;
        }
        if (!a.isContiguous() || !b.isContiguous() || !out.isContiguous()) {
            return false;
        }
        if (blasConfig.provider() != BlasProvider.OPENBLAS_FFM) {
            return false;
        }
        if (attentionMatMulPolicy == AttentionMatMulPolicy.FORCE_OFF) {
            return false;
        }
        if (attentionMatMulPolicy == AttentionMatMulPolicy.FORCE_ON) {
            return true;
        }
        if (work < blasConfig.matMulMinWork()) {
            return false;
        }
        if (out.getDataType() == DataType.FLOAT32 || out.getDataType() == DataType.BFLOAT16) {
            if (blasConfig.f32RequireMgeK() && m < k) {
                return false;
            }
            if (((double) n / Math.max(1, k)) > blasConfig.f32MaxNOverK()) {
                return false;
            }
        }
        return true;
    }

    private static boolean isAttentionLikeBatchedMatMul(Tensor a, Tensor b, Tensor out) {
        int[] as = a.getShapeUnsafe();
        int[] bs = b.getShapeUnsafe();
        int[] os = out.getShapeUnsafe();
        if (as.length < 3 || bs.length < 3 || os.length < 3) {
            return false;
        }
        boolean scoreLike = os[os.length - 2] == bs[bs.length - 1];
        boolean weightsValueLike = as[as.length - 2] == as[as.length - 1];
        return scoreLike || weightsValueLike;
    }

    private VectorPolicy resolveElementWisePolicy(Operation op) {
        if (op == null || op.opType() == null) {
            return vectorPolicyCheap;
        }
        return switch (op.opType()) {
            case LOG, EXP, FAST_EXP, TANH, FAST_TANH, POW, SQRT, SIGMOID -> vectorPolicyTranscendental;
            case FUSED -> {
                if (op instanceof FusedOperation fused) {
                    yield fused.isLowCostHint() ? vectorPolicyCheap : vectorPolicyTranscendental;
                }
                yield vectorPolicyTranscendental;
            }
            default -> vectorPolicyCheap;
        };
    }

    private boolean isVectorAllowed(VectorPolicy policy, int size, int effectiveVectorMinSize) {
        return switch (policy) {
            case FORCE_ON -> true;
            case FORCE_OFF -> false;
            case AUTO -> size >= Math.max(1, effectiveVectorMinSize);
        };
    }

    private int effectiveVectorMinSize(Operation op) {
        int base = Math.max(1, vectorMinSize);
        if (op instanceof FusedOperation fused) {
            int scale = Math.max(1, fused.getDispatchScale());
            if (scale > 1) {
                return Math.max(1, base / scale);
            }
        }
        return base;
    }

    private int effectiveParallelMinSize(Operation op) {
        int base = Math.max(1, parallelMinSize);
        if (op instanceof FusedOperation fused) {
            int scale = Math.max(1, fused.getDispatchScale());
            if (scale > 1) {
                return Math.max(1, base / scale);
            }
        }
        return base;
    }

    private static int positiveOrDefault(int value, int fallback) {
        return value > 0 ? value : fallback;
    }
}
