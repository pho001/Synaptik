package backend.kernels.cpu;

import backend.blas.BlasProvider;
import backend.runtime.BlasConfig;
import config.backend.AttentionMatMulPolicy;
import config.backend.CpuMatMulMicroKernel;
import config.backend.CpuKernelConfig;
import config.backend.SumAccuracyMode;
import graph.codegen.FusedAsmSpecializationKind;
import graph.codegen.FusedAsmSpecializationMatcher;
import graph.codegen.FusedAccessKind;
import graph.codegen.FusedExternalInputPlan;
import graph.codegen.FusedNodePlan;
import graph.optimizer.fusion.FusedDispatchFamily;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import operations.FusedOperation;
import operations.Operation;
import tensor.DataType;
import tensor.Tensor;

import java.util.List;
import java.util.Objects;

public final class CpuExecutionPlanner {
    public static final int DEFAULT_MATMUL_TILE_M = 32;
    public static final int DEFAULT_MATMUL_TILE_N = 64;
    public static final int DEFAULT_MATMUL_TILE_K = 64;

    private final int matMulTileM;
    private final int matMulTileN;
    private final int matMulTileK;
    private final int cheapVectorMinSize;
    private final int transcendentalVectorMinSize;
    private final int fusedCheapVectorMinSize;
    private final int fusedTranscendentalVectorMinSize;
    private final int reductionVectorMinSize;
    private final int cheapParallelMinSize;
    private final int transcendentalParallelMinSize;
    private final int fusedCheapParallelMinSize;
    private final int fusedTranscendentalParallelMinSize;
    private final int reductionParallelMinSize;
    private final int matMulParallelMinSize;
    private final int contiguousMaterializeThreshold;
    private final int lowCostTargetChunksPerWorker;
    private final int mediumCostTargetChunksPerWorker;
    private final int highCostTargetChunksPerWorker;
    private final int minScalarChunkSize;
    private final int minVectorChunkSize;
    private final int minReductionChunkSize;
    private final int commonPoolLowCostMaxWorkPerWorker;
    private final int fusedCheapContiguousAsmVectorWidth;
    private final int fusedCheapStridedAsmVectorWidth;
    private final int fusedNonCheapContiguousAsmVectorWidth;
    private final int fusedNonCheapStridedAsmVectorWidth;
    private final SumAccuracyMode sumAccuracyMode;
    private final AttentionMatMulPolicy attentionMatMulPolicy;
    private final CpuMatMulMicroKernel matMulMicroKernel;

    public CpuExecutionPlanner(
            int matMulTileM,
            int matMulTileN,
            int matMulTileK,
            int cheapVectorMinSize,
            int transcendentalVectorMinSize,
            int fusedCheapVectorMinSize,
            int fusedTranscendentalVectorMinSize,
            int reductionVectorMinSize,
            int cheapParallelMinSize,
            int transcendentalParallelMinSize,
            int fusedCheapParallelMinSize,
            int fusedTranscendentalParallelMinSize,
            int reductionParallelMinSize,
            int matMulParallelMinSize,
            int contiguousMaterializeThreshold,
            int lowCostTargetChunksPerWorker,
            int mediumCostTargetChunksPerWorker,
            int highCostTargetChunksPerWorker,
            int minScalarChunkSize,
            int minVectorChunkSize,
            int minReductionChunkSize,
            int commonPoolLowCostMaxWorkPerWorker,
            int fusedCheapContiguousAsmVectorWidth,
            int fusedCheapStridedAsmVectorWidth,
            int fusedNonCheapContiguousAsmVectorWidth,
            int fusedNonCheapStridedAsmVectorWidth,
            SumAccuracyMode sumAccuracyMode,
            AttentionMatMulPolicy attentionMatMulPolicy,
            CpuMatMulMicroKernel matMulMicroKernel
    ) {
        this.matMulTileM = positiveOrDefault(matMulTileM, DEFAULT_MATMUL_TILE_M);
        this.matMulTileN = positiveOrDefault(matMulTileN, DEFAULT_MATMUL_TILE_N);
        this.matMulTileK = positiveOrDefault(matMulTileK, DEFAULT_MATMUL_TILE_K);
        this.cheapVectorMinSize = Math.max(1, cheapVectorMinSize);
        this.transcendentalVectorMinSize = Math.max(1, transcendentalVectorMinSize);
        this.fusedCheapVectorMinSize = Math.max(1, fusedCheapVectorMinSize);
        this.fusedTranscendentalVectorMinSize = Math.max(1, fusedTranscendentalVectorMinSize);
        this.reductionVectorMinSize = Math.max(1, reductionVectorMinSize);
        this.cheapParallelMinSize = Math.max(1, cheapParallelMinSize);
        this.transcendentalParallelMinSize = Math.max(1, transcendentalParallelMinSize);
        this.fusedCheapParallelMinSize = Math.max(1, fusedCheapParallelMinSize);
        this.fusedTranscendentalParallelMinSize = Math.max(1, fusedTranscendentalParallelMinSize);
        this.reductionParallelMinSize = Math.max(1, reductionParallelMinSize);
        this.matMulParallelMinSize = Math.max(1, matMulParallelMinSize);
        this.contiguousMaterializeThreshold = Math.max(0, contiguousMaterializeThreshold);
        this.lowCostTargetChunksPerWorker = Math.max(1, lowCostTargetChunksPerWorker);
        this.mediumCostTargetChunksPerWorker = Math.max(1, mediumCostTargetChunksPerWorker);
        this.highCostTargetChunksPerWorker = Math.max(1, highCostTargetChunksPerWorker);
        this.minScalarChunkSize = Math.max(1, minScalarChunkSize);
        this.minVectorChunkSize = Math.max(1, minVectorChunkSize);
        this.minReductionChunkSize = Math.max(1, minReductionChunkSize);
        this.commonPoolLowCostMaxWorkPerWorker = Math.max(1, commonPoolLowCostMaxWorkPerWorker);
        this.fusedCheapContiguousAsmVectorWidth = Math.max(1, fusedCheapContiguousAsmVectorWidth);
        this.fusedCheapStridedAsmVectorWidth = Math.max(1, fusedCheapStridedAsmVectorWidth);
        this.fusedNonCheapContiguousAsmVectorWidth = Math.max(1, fusedNonCheapContiguousAsmVectorWidth);
        this.fusedNonCheapStridedAsmVectorWidth = Math.max(1, fusedNonCheapStridedAsmVectorWidth);
        this.sumAccuracyMode = Objects.requireNonNullElse(sumAccuracyMode, SumAccuracyMode.FAST);
        this.attentionMatMulPolicy = Objects.requireNonNullElse(attentionMatMulPolicy, AttentionMatMulPolicy.AUTO);
        this.matMulMicroKernel = Objects.requireNonNullElse(matMulMicroKernel, CpuMatMulMicroKernel.AUTO);
    }

    public static CpuExecutionPlanner from(CpuKernelConfig config) {
        Objects.requireNonNull(config, "config cannot be null");
        return new CpuExecutionPlanner(
                config.matMulTileM(),
                config.matMulTileN(),
                config.matMulTileK(),
                config.cheapVectorMinSize(),
                config.transcendentalVectorMinSize(),
                config.fusedCheapVectorMinSize(),
                config.fusedTranscendentalVectorMinSize(),
                config.reductionVectorMinSize(),
                config.cheapParallelMinSize(),
                config.transcendentalParallelMinSize(),
                config.fusedCheapParallelMinSize(),
                config.fusedTranscendentalParallelMinSize(),
                config.reductionParallelMinSize(),
                config.matMulParallelMinSize(),
                config.contiguousMaterializeThreshold(),
                config.lowCostTargetChunksPerWorker(),
                config.mediumCostTargetChunksPerWorker(),
                config.highCostTargetChunksPerWorker(),
                config.minScalarChunkSize(),
                config.minVectorChunkSize(),
                config.minReductionChunkSize(),
                config.commonPoolLowCostMaxWorkPerWorker(),
                config.fusedCheapContiguousAsmVectorWidth(),
                config.fusedCheapStridedAsmVectorWidth(),
                config.fusedNonCheapContiguousAsmVectorWidth(),
                config.fusedNonCheapStridedAsmVectorWidth(),
                config.sumAccuracyMode(),
                config.attentionMatMulPolicy(),
                config.matMulMicroKernel()
        );
    }

    public int contiguousMaterializeThreshold() {
        return contiguousMaterializeThreshold;
    }

    public SumAccuracyMode sumAccuracyMode() {
        return sumAccuracyMode;
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
        return Math.max(1, Runtime.getRuntime().availableProcessors());
    }

    public boolean shouldMaterializeNonContiguous(int logicalSize) {
        return logicalSize >= contiguousMaterializeThreshold;
    }

    public int preferredVectorWidth(ResolvedCpuComputeContract contract) {
        if (contract == null) {
            return 1;
        }
        return switch (contract.computeType()) {
            case F64 -> DoubleVector.SPECIES_PREFERRED.length();
            case F32, BF16_NATIVE -> FloatVector.SPECIES_PREFERRED.length();
            case INT32, BOOL -> 1;
        };
    }

    public int resolvedFusedAsmVectorWidth(ResolvedCpuComputeContract contract, FusedOperation fused) {
        if (contract == null) {
            return 1;
        }
        if (shouldForceScalarFusedAsm(fused)) {
            return 1;
        }
        int available = switch (contract.computeType()) {
            case F32, BF16_NATIVE -> FloatVector.SPECIES_PREFERRED.length();
            case F64 -> DoubleVector.SPECIES_PREFERRED.length();
            case INT32, BOOL -> 1;
        };
        int configuredWidth = resolveFusedAsmVectorWidthForFamily(
                fused == null ? FusedDispatchFamily.NON_CHEAP_STRIDED : fused.getDispatchFamily()
        );
        if (configuredWidth <= 1 || available <= 1) {
            return 1;
        }
        int width = Math.min(configuredWidth, available);
        if (width >= 8) {
            return 8;
        }
        if (width >= 4) {
            return 4;
        }
        if (width >= 2) {
            return 2;
        }
        return 1;
    }

    private boolean shouldForceScalarFusedAsm(FusedOperation fused) {
        if (fused == null || fused.getPlan() == null) {
            return false;
        }
        FusedAsmSpecializationKind specializationKind =
                FusedAsmSpecializationMatcher.match(fused.getPlan(), fused.getPrecisionMode());
        return specializationKind == FusedAsmSpecializationKind.F32_MASKED_SCALE_WHERE
                || specializationKind == FusedAsmSpecializationKind.F32_MASKED_SCALE_WHERE_INVERTED;
    }

    private boolean shouldForceSerialScalarDispatch(FusedOperation fused) {
        return shouldForceScalarFusedAsm(fused);
    }

    public ResolvedDispatchHints resolveDispatchHints(Operation op, Tensor node, ResolvedCpuComputeContract contract) {
        if (op == null || node == null
                || (op.opType().category() != Operation.OpArityClass.ELEMENT_WISE && op.opType() != Operation.OpType.FUSED)) {
            return new ResolvedDispatchHints(0, CpuExecutionMode.SCALAR, 1, 1, 1, 1, false);
        }

        int totalLength = Math.max(0, node.getFlatDataSize());
        boolean fused = op.opType() == Operation.OpType.FUSED;
        CpuKernelCostClass costClass = resolveDispatchCostClass(op);
        if (fused && shouldForceSerialScalarDispatch((FusedOperation) op)) {
            return new ResolvedDispatchHints(
                    totalLength,
                    CpuExecutionMode.SCALAR,
                    computeChunkSize(totalLength, 1, resolveTargetChunksPerWorker(costClass), minScalarChunkSize),
                    computeChunkSize(totalLength, 1, resolveTargetChunksPerWorker(costClass), minVectorChunkSize),
                    1,
                    1,
                    false
            );
        }
        int vectorWidth = fused ? resolvedFusedAsmVectorWidth(contract, (FusedOperation) op) : preferredVectorWidth(contract);
        boolean vectorAllowed = vectorWidth > 1 && totalLength >= effectiveVectorMinSize(op);

        CpuExecutionMode mode;
        if (totalLength >= effectiveParallelMinSize(op)) {
            mode = vectorAllowed ? CpuExecutionMode.PARALLEL_VECTOR : CpuExecutionMode.PARALLEL;
        } else {
            mode = vectorAllowed ? CpuExecutionMode.VECTOR : CpuExecutionMode.SCALAR;
        }

        return new ResolvedDispatchHints(
                totalLength,
                mode,
                computeChunkSize(totalLength, 1, resolveTargetChunksPerWorker(costClass), minScalarChunkSize),
                computeChunkSize(totalLength, vectorWidth, resolveTargetChunksPerWorker(costClass), minVectorChunkSize),
                vectorWidth,
                plannedWorkers(),
                (mode == CpuExecutionMode.PARALLEL || mode == CpuExecutionMode.PARALLEL_VECTOR)
                        && shouldUseCommonPool(costClass, totalLength)
        );
    }

    public ResolvedReductionHints resolveReductionHints(int logicalSize, ResolvedCpuComputeContract contract) {
        int size = Math.max(0, logicalSize);
        int vectorWidth = preferredVectorWidth(contract);
        boolean vectorAllowed = vectorWidth > 1 && size >= reductionVectorMinSize;

        CpuExecutionMode mode;
        if (size >= reductionParallelMinSize) {
            mode = vectorAllowed ? CpuExecutionMode.PARALLEL_VECTOR : CpuExecutionMode.PARALLEL;
        } else {
            mode = vectorAllowed ? CpuExecutionMode.VECTOR : CpuExecutionMode.SCALAR;
        }

        int chunkSize = computeChunkSize(
                size,
                mode == CpuExecutionMode.VECTOR || mode == CpuExecutionMode.PARALLEL_VECTOR ? vectorWidth : 1,
                1,
                minReductionChunkSize
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

    public ResolvedCpuComputeContract resolveComputeContract(
            Operation op,
            List<Tensor> inputs,
            Tensor node,
            BlasConfig blasConfig,
            ResolvedMatMulHints matMulHints
    ) {
        Objects.requireNonNull(node, "node cannot be null");
        DataType dataType = node.getDataType() == null ? DataType.FLOAT64 : node.getDataType();
        if (op == null) {
            return defaultContractFor(dataType, CpuExecutionBackend.CPU_GENERIC);
        }
        return switch (op.opType()) {
            case MATMUL, LINEAR -> resolveMatMulContract(dataType, matMulHints);
            case SUM, MEAN, REDUCE_MIN, REDUCE_MAX, REDUCE_ALL, REDUCE_ANY, SOFTMAX, LOG_SOFTMAX, NLL_LOSS, CROSS_ENTROPY_LOSS, CROSS_ENTROPY_LOSS_INDICES, CROSS_ENTROPY_LOSS_INDICES_GRAD ->
                    resolveReductionContract(dataType);
            case FUSED -> defaultContractFor(dataType, CpuExecutionBackend.CPU_FUSED);
            default -> (op.opType().category() == Operation.OpArityClass.ELEMENT_WISE)
                    ? defaultContractFor(dataType, CpuExecutionBackend.CPU_ELEMENTWISE)
                    : defaultContractFor(dataType, CpuExecutionBackend.CPU_GENERIC);
        };
    }

    private ResolvedCpuComputeContract resolveMatMulContract(DataType dataType, ResolvedMatMulHints matMulHints) {
        CpuExecutionBackend backend = (matMulHints != null && (matMulHints.useBlas() || matMulHints.useBatchedBlas()))
                ? CpuExecutionBackend.CPU_MATMUL_BLAS
                : CpuExecutionBackend.CPU_MATMUL_JAVA;
        return switch (dataType) {
            case FLOAT64 -> new ResolvedCpuComputeContract(dataType, CpuComputeDType.F64, backend, CpuAccumulateDType.NONE);
            case FLOAT32 -> new ResolvedCpuComputeContract(dataType, CpuComputeDType.F32, backend, CpuAccumulateDType.NONE);
            case BFLOAT16 -> new ResolvedCpuComputeContract(dataType, CpuComputeDType.F32, backend, CpuAccumulateDType.NONE);
            case INT32 -> new ResolvedCpuComputeContract(dataType, CpuComputeDType.INT32, backend, CpuAccumulateDType.NONE);
            case BOOL -> new ResolvedCpuComputeContract(dataType, CpuComputeDType.BOOL, backend, CpuAccumulateDType.NONE);
        };
    }

    private ResolvedCpuComputeContract resolveReductionContract(DataType dataType) {
        return switch (dataType) {
            case FLOAT64 -> new ResolvedCpuComputeContract(dataType, CpuComputeDType.F64, CpuExecutionBackend.CPU_REDUCTION, CpuAccumulateDType.F64);
            case FLOAT32 -> new ResolvedCpuComputeContract(dataType, CpuComputeDType.F32, CpuExecutionBackend.CPU_REDUCTION, CpuAccumulateDType.F64);
            case BFLOAT16 -> new ResolvedCpuComputeContract(dataType, CpuComputeDType.F32, CpuExecutionBackend.CPU_REDUCTION, CpuAccumulateDType.F64);
            case INT32 -> new ResolvedCpuComputeContract(dataType, CpuComputeDType.INT32, CpuExecutionBackend.CPU_REDUCTION, CpuAccumulateDType.NONE);
            case BOOL -> new ResolvedCpuComputeContract(dataType, CpuComputeDType.BOOL, CpuExecutionBackend.CPU_REDUCTION, CpuAccumulateDType.NONE);
        };
    }

    private ResolvedCpuComputeContract defaultContractFor(DataType dataType, CpuExecutionBackend backend) {
        return switch (dataType) {
            case FLOAT64 -> new ResolvedCpuComputeContract(dataType, CpuComputeDType.F64, backend, CpuAccumulateDType.NONE);
            case FLOAT32 -> new ResolvedCpuComputeContract(dataType, CpuComputeDType.F32, backend, CpuAccumulateDType.NONE);
            case BFLOAT16 -> new ResolvedCpuComputeContract(dataType, CpuComputeDType.F32, backend, CpuAccumulateDType.NONE);
            case INT32 -> new ResolvedCpuComputeContract(dataType, CpuComputeDType.INT32, backend, CpuAccumulateDType.NONE);
            case BOOL -> new ResolvedCpuComputeContract(dataType, CpuComputeDType.BOOL, backend, CpuAccumulateDType.NONE);
        };
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
                work,
                matMulMicroKernel.resolve(out.getDataType())
        );
    }

    public int computeChunkSize(int totalLength, int alignment, int targetChunksPerWorker, int minChunkSize) {
        int length = Math.max(1, totalLength);
        int workers = plannedWorkers();
        int targets = Math.max(workers, workers * Math.max(1, targetChunksPerWorker));
        int candidate = (length + targets - 1) / targets;
        int chunk = Math.max(Math.max(1, minChunkSize), candidate);

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
        return switch (attentionMatMulPolicy) {
            case FORCE_OFF -> false;
            case FORCE_ON -> true;
            case AUTO -> true;
        };
    }

    private boolean isAttentionLikeBatchedMatMul(Tensor a, Tensor b, Tensor out) {
        int[] as = a.getShapeUnsafe();
        int[] bs = b.getShapeUnsafe();
        int[] os = out.getShapeUnsafe();
        if (as.length < 3 || bs.length < 3 || os.length < 3) {
            return false;
        }
        int aBatchRank = as.length - 2;
        int bBatchRank = bs.length - 2;
        int oBatchRank = os.length - 2;
        if (aBatchRank != bBatchRank || aBatchRank != oBatchRank) {
            return false;
        }
        if (aBatchRank < 1) {
            return false;
        }
        if (oBatchRank == 1) {
            return false;
        }
        return os[oBatchRank - 1] == as[aBatchRank - 1];
    }

    private int effectiveVectorMinSize(Operation op) {
        int base = Math.max(1, resolveBaseVectorMinSize(op));
        if (op == null) {
            return base;
        }
        if (op.opType() == Operation.OpType.FUSED) {
            FusedOperation fused = (FusedOperation) op;
            int adjustedBase = adjustFusedVectorMinSize(base, fused);
            int scale = fused.getDispatchScale();
            return Math.max(1, adjustedBase / Math.max(1, scale));
        }
        return base;
    }

    public int fusedDirectVectorMinSize(FusedOperation operation) {
        return effectiveVectorMinSize(operation);
    }

    private int effectiveParallelMinSize(Operation op) {
        int base = Math.max(1, resolveBaseParallelMinSize(op));
        if (op == null) {
            return base;
        }
        if (op.opType() == Operation.OpType.FUSED) {
            int scale = ((FusedOperation) op).getDispatchScale();
            return Math.max(1, base / Math.max(1, scale));
        }
        return base;
    }

    private int resolveBaseParallelMinSize(Operation op) {
        if (op == null) {
            return cheapParallelMinSize;
        }
        if (op.opType() == Operation.OpType.FUSED && op instanceof FusedOperation fused) {
            return fusedContainsTranscendental(fused) ? fusedTranscendentalParallelMinSize : fusedCheapParallelMinSize;
        }
        return switch (op.opType()) {
            case EXP, FAST_EXP, TANH, FAST_TANH, LOG, SIGMOID, POW -> transcendentalParallelMinSize;
            default -> cheapParallelMinSize;
        };
    }

    private CpuKernelCostClass resolveDispatchCostClass(Operation op) {
        if (op == null) {
            return CpuKernelCostClass.MEDIUM;
        }
        if (op instanceof FusedOperation fused) {
            return fused.isLowCostHint() && fused.getDispatchScale() == 1
                    ? CpuKernelCostClass.LOW
                    : CpuKernelCostClass.MEDIUM;
        }
        return op.isCheap() ? CpuKernelCostClass.LOW : CpuKernelCostClass.MEDIUM;
    }

    private int resolveBaseVectorMinSize(Operation op) {
        if (op == null) {
            return cheapVectorMinSize;
        }
        if (op.opType() == Operation.OpType.FUSED && op instanceof FusedOperation fused) {
            return fusedContainsTranscendental(fused) ? fusedTranscendentalVectorMinSize : fusedCheapVectorMinSize;
        }
        return switch (op.opType()) {
            case EXP, FAST_EXP, TANH, FAST_TANH, LOG, SIGMOID, POW -> transcendentalVectorMinSize;
            default -> cheapVectorMinSize;
        };
    }

    private int adjustFusedVectorMinSize(int base, FusedOperation fused) {
        if (fused == null || fused.getPlan() == null) {
            return base;
        }
        int dispatchScale = Math.max(1, fused.getDispatchScale());
        if ((fused.getDispatchFamily() == FusedDispatchFamily.CHEAP_CONTIGUOUS
                || fused.getDispatchFamily() == FusedDispatchFamily.CHEAP_STRIDED)
                && fused.getPlan().nodeCount() <= 2) {
            return conservativeFusedVectorMinSize(base, dispatchScale);
        }
        if (fused.getDispatchFamily() == FusedDispatchFamily.NON_CHEAP_STRIDED
                && !isVectorFriendlyNonCheapStridedPlan(fused)) {
            return conservativeFusedVectorMinSize(base, dispatchScale);
        }
        return base;
    }

    private boolean isVectorFriendlyNonCheapStridedPlan(FusedOperation fused) {
        if (fused == null || fused.getPlan() == null) {
            return false;
        }
        if (fusedContainsTranscendental(fused)) {
            return false;
        }
        boolean hasWhere = false;
        boolean hasBoolInput = false;
        boolean hasBroadcastInput = false;
        for (FusedNodePlan node : fused.getPlan().nodes()) {
            if (node.opType() == Operation.OpType.WHERE) {
                hasWhere = true;
            }
        }
        for (FusedExternalInputPlan input : fused.getPlan().inputs()) {
            if (input.dataType() == DataType.BOOL) {
                hasBoolInput = true;
            }
            if (input.accessKind() == FusedAccessKind.BROADCAST_STRIDED) {
                hasBroadcastInput = true;
            }
        }
        return hasWhere && hasBoolInput && hasBroadcastInput;
    }

    private boolean fusedContainsTranscendental(FusedOperation fused) {
        if (fused == null || fused.getPlan() == null) {
            return false;
        }
        return fused.getPlan().nodes().stream().anyMatch(node -> switch (node.opType()) {
            case EXP, FAST_EXP, TANH, FAST_TANH, LOG, SIGMOID, POW -> true;
            default -> false;
        });
    }

    private int resolveTargetChunksPerWorker(CpuKernelCostClass costClass) {
        return switch (costClass) {
            case LOW -> lowCostTargetChunksPerWorker;
            case MEDIUM -> mediumCostTargetChunksPerWorker;
            case HIGH -> highCostTargetChunksPerWorker;
        };
    }

    private boolean shouldUseCommonPool(CpuKernelCostClass costClass, int totalLength) {
        if (costClass != CpuKernelCostClass.LOW) {
            return false;
        }
        return totalLength <= (long) plannedWorkers() * commonPoolLowCostMaxWorkPerWorker;
    }

    private int resolveFusedAsmVectorWidthForFamily(FusedDispatchFamily family) {
        if (family == null) {
            return fusedNonCheapStridedAsmVectorWidth;
        }
        return switch (family) {
            case CHEAP_CONTIGUOUS -> fusedCheapContiguousAsmVectorWidth;
            case CHEAP_STRIDED -> fusedCheapStridedAsmVectorWidth;
            case NON_CHEAP_CONTIGUOUS -> fusedNonCheapContiguousAsmVectorWidth;
            case NON_CHEAP_STRIDED -> fusedNonCheapStridedAsmVectorWidth;
        };
    }

    private static int saturatingMultiply(int value, int factor) {
        long product = (long) Math.max(1, value) * Math.max(1, factor);
        return product >= Integer.MAX_VALUE ? Integer.MAX_VALUE : (int) product;
    }

    private int conservativeFusedVectorMinSize(int base, int dispatchScale) {
        int conservativeMultiplier = saturatingMultiply(16, dispatchScale);
        return Math.max(
                saturatingMultiply(base, conservativeMultiplier),
                saturatingMultiply(minVectorChunkSize, conservativeMultiplier)
        );
    }

    private static int positiveOrDefault(int value, int defaultValue) {
        return value > 0 ? value : defaultValue;
    }
}
