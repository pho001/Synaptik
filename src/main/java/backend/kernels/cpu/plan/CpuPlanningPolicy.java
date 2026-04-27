package backend.kernels.cpu.plan;

import config.backend.AttentionMatMulPolicy;
import config.backend.CpuMatMulMicroKernel;
import config.backend.SumAccuracyMode;
import backend.cpu.fused.codegen.FusedAsmSpecializationKind;
import backend.cpu.fused.codegen.FusedAsmSpecializationMatcher;
import backend.cpu.fused.codegen.FusedAccessKind;
import backend.cpu.fused.codegen.FusedExternalInputPlan;
import backend.cpu.fused.codegen.FusedNodePlan;
import backend.cpu.fused.optimize.FusedDispatchFamily;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import backend.cpu.fused.plan.FusedOperation;
import operations.Operation;
import backend.kernels.cpu.CpuExecutionBackend;
import backend.kernels.cpu.CpuKernelCostClass;
import backend.kernels.cpu.CpuComputeDType;
import backend.kernels.cpu.ResolvedCpuComputeContract;
import tensor.DataType;
import backend.cpu.fused.codegen.FusedDTypeOps;

import java.util.Objects;

public final class CpuPlanningPolicy {
    public static final int DEFAULT_MATMUL_TILE_M = 32;
    public static final int DEFAULT_MATMUL_TILE_N = 64;
    public static final int DEFAULT_MATMUL_TILE_K = 64;

    private final int matMulTileM;
    private final int matMulTileN;
    private final int matMulTileK;
    private final int attentionMatMulTileM;
    private final int attentionMatMulTileN;
    private final int attentionMatMulTileK;
    private final int cheapVectorMinSize;
    private final int transcendentalVectorMinSize;
    private final int fusedCheapVectorMinSize;
    private final int fusedTranscendentalVectorMinSize;
    private final int reductionVectorMinSize;
    private final int attentionVectorMinSize;
    private final int cheapParallelMinSize;
    private final int transcendentalParallelMinSize;
    private final int fusedCheapParallelMinSize;
    private final int fusedTranscendentalParallelMinSize;
    private final int reductionParallelMinSize;
    private final int attentionParallelMinSize;
    private final int matMulParallelMinSize;
    private final int contiguousMaterializeThreshold;
    private final int cheapF64MaterializeThreshold;
    private final int cheapF32MaterializeThreshold;
    private final int cheapBF16MaterializeThreshold;
    private final int whereMaterializeThreshold;
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
    private final CpuMatMulMicroKernel attentionMatMulMicroKernel;

    public CpuPlanningPolicy(
            int matMulTileM,
            int matMulTileN,
            int matMulTileK,
            int cheapVectorMinSize,
            int transcendentalVectorMinSize,
            int fusedCheapVectorMinSize,
            int fusedTranscendentalVectorMinSize,
            int reductionVectorMinSize,
            int attentionVectorMinSize,
            int cheapParallelMinSize,
            int transcendentalParallelMinSize,
            int fusedCheapParallelMinSize,
            int fusedTranscendentalParallelMinSize,
            int reductionParallelMinSize,
            int attentionParallelMinSize,
            int matMulParallelMinSize,
            int contiguousMaterializeThreshold,
            int cheapF64MaterializeThreshold,
            int cheapF32MaterializeThreshold,
            int cheapBF16MaterializeThreshold,
            int whereMaterializeThreshold,
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
            CpuMatMulMicroKernel matMulMicroKernel,
            CpuMatMulMicroKernel attentionMatMulMicroKernel,
            int attentionMatMulTileM,
            int attentionMatMulTileN,
            int attentionMatMulTileK
    ) {
        this.matMulTileM = positiveOrDefault(matMulTileM, DEFAULT_MATMUL_TILE_M);
        this.matMulTileN = positiveOrDefault(matMulTileN, DEFAULT_MATMUL_TILE_N);
        this.matMulTileK = positiveOrDefault(matMulTileK, DEFAULT_MATMUL_TILE_K);
        this.attentionMatMulTileM = positiveOrDefault(attentionMatMulTileM, this.matMulTileM);
        this.attentionMatMulTileN = positiveOrDefault(attentionMatMulTileN, this.matMulTileN);
        this.attentionMatMulTileK = positiveOrDefault(attentionMatMulTileK, this.matMulTileK);
        this.cheapVectorMinSize = Math.max(1, cheapVectorMinSize);
        this.transcendentalVectorMinSize = Math.max(1, transcendentalVectorMinSize);
        this.fusedCheapVectorMinSize = Math.max(1, fusedCheapVectorMinSize);
        this.fusedTranscendentalVectorMinSize = Math.max(1, fusedTranscendentalVectorMinSize);
        this.reductionVectorMinSize = Math.max(1, reductionVectorMinSize);
        this.attentionVectorMinSize = Math.max(1, attentionVectorMinSize);
        this.cheapParallelMinSize = Math.max(1, cheapParallelMinSize);
        this.transcendentalParallelMinSize = Math.max(1, transcendentalParallelMinSize);
        this.fusedCheapParallelMinSize = Math.max(1, fusedCheapParallelMinSize);
        this.fusedTranscendentalParallelMinSize = Math.max(1, fusedTranscendentalParallelMinSize);
        this.reductionParallelMinSize = Math.max(1, reductionParallelMinSize);
        this.attentionParallelMinSize = Math.max(1, attentionParallelMinSize);
        this.matMulParallelMinSize = Math.max(1, matMulParallelMinSize);
        this.contiguousMaterializeThreshold = Math.max(0, contiguousMaterializeThreshold);
        this.cheapF64MaterializeThreshold = Math.max(0, cheapF64MaterializeThreshold);
        this.cheapF32MaterializeThreshold = Math.max(0, cheapF32MaterializeThreshold);
        this.cheapBF16MaterializeThreshold = Math.max(0, cheapBF16MaterializeThreshold);
        this.whereMaterializeThreshold = Math.max(0, whereMaterializeThreshold);
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
        this.attentionMatMulMicroKernel = Objects.requireNonNullElse(attentionMatMulMicroKernel, this.matMulMicroKernel);
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

    public int attentionMatMulTileM() {
        return attentionMatMulTileM;
    }

    public int attentionMatMulTileN() {
        return attentionMatMulTileN;
    }

    public int attentionMatMulTileK() {
        return attentionMatMulTileK;
    }

    public int matMulParallelMinSize() {
        return matMulParallelMinSize;
    }

    public int reductionVectorMinSize() {
        return reductionVectorMinSize;
    }

    public int attentionVectorMinSize() {
        return attentionVectorMinSize;
    }

    public int reductionParallelMinSize() {
        return reductionParallelMinSize;
    }

    public int attentionParallelMinSize() {
        return attentionParallelMinSize;
    }

    public AttentionMatMulPolicy attentionMatMulPolicy() {
        return attentionMatMulPolicy;
    }

    public CpuMatMulMicroKernel matMulMicroKernel() {
        return matMulMicroKernel;
    }

    public CpuMatMulMicroKernel attentionMatMulMicroKernel() {
        return attentionMatMulMicroKernel;
    }

    public int plannedWorkers() {
        return Math.max(1, Runtime.getRuntime().availableProcessors());
    }

    public boolean shouldMaterializeNonContiguous(int logicalSize) {
        return logicalSize >= contiguousMaterializeThreshold;
    }

    public int cheapElementwiseMaterializeThreshold(DataType targetType) {
        if (targetType == null) {
            return contiguousMaterializeThreshold;
        }
        return switch (targetType) {
            case FLOAT64 -> cheapF64MaterializeThreshold;
            case FLOAT32 -> cheapF32MaterializeThreshold;
            case BFLOAT16 -> cheapBF16MaterializeThreshold;
            case BOOL, INT32 -> contiguousMaterializeThreshold;
        };
    }

    public int whereMaterializeThreshold() {
        return whereMaterializeThreshold;
    }

    public boolean shouldMaterializeCheapStridedElementwise(Operation op, DataType targetType, int logicalSize) {
        if (op == null || logicalSize <= 0) {
            return false;
        }
        if (op.opType() == Operation.OpType.WHERE) {
            return logicalSize >= whereMaterializeThreshold;
        }
        if (!isCheapElementwiseMaterializationCandidate(op)) {
            return false;
        }
        return logicalSize >= cheapElementwiseMaterializeThreshold(targetType);
    }

    private boolean isCheapElementwiseMaterializationCandidate(Operation op) {
        if (op == null || op.opType() == null) {
            return false;
        }
        return switch (op.opType()) {
            case ADD, SUB, MUL, DIV, MIN, MAX,
                    GT, GE, LT, LE, EQ, NE,
                    LOGICAL_AND, LOGICAL_OR, LOGICAL_NOT,
                    NEG, INV, ABS, MUL_SCALAR, RELU, CLAMP_MIN, CLAMP_MAX -> true;
            default -> false;
        };
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
        if (shouldForceScalarFusedAsm(fused) || shouldClampBf16NonCheapStridedAsmWidthToScalar(contract, fused)) {
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

    public boolean shouldForceSerialScalarDispatch(FusedOperation fused) {
        return shouldForceScalarFusedAsm(fused);
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

    public int elementwiseVectorMinSize(Operation operation) {
        return effectiveVectorMinSize(operation);
    }

    public int elementwiseParallelMinSize(Operation operation) {
        return effectiveParallelMinSize(operation);
    }

    public int fusedDirectVectorMinSize(FusedOperation operation) {
        return effectiveVectorMinSize(operation);
    }

    public int fusedParallelMinSize(FusedOperation operation) {
        return effectiveParallelMinSize(operation);
    }

    public CpuKernelCostClass dispatchCostClass(Operation op) {
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

    public int targetChunksPerWorker(CpuKernelCostClass costClass) {
        return switch (costClass) {
            case LOW -> lowCostTargetChunksPerWorker;
            case MEDIUM -> mediumCostTargetChunksPerWorker;
            case HIGH -> highCostTargetChunksPerWorker;
        };
    }

    public boolean shouldUseCommonPoolFor(CpuKernelCostClass costClass, int totalLength) {
        if (costClass != CpuKernelCostClass.LOW) {
            return false;
        }
        return totalLength <= (long) plannedWorkers() * commonPoolLowCostMaxWorkPerWorker;
    }

    public int minScalarChunkSize() {
        return minScalarChunkSize;
    }

    public int minVectorChunkSize() {
        return minVectorChunkSize;
    }

    public int minReductionChunkSize() {
        return minReductionChunkSize;
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

    private boolean shouldClampBf16NonCheapStridedAsmWidthToScalar(
            ResolvedCpuComputeContract contract,
            FusedOperation fused
    ) {
        if (contract == null
                || contract.computeType() != CpuComputeDType.BF16_NATIVE
                || fused == null
                || fused.getDispatchFamily() != FusedDispatchFamily.NON_CHEAP_STRIDED) {
            return false;
        }
        return isBf16AffineRationalStridedPlan(fused);
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

    private boolean isBf16AffineRationalStridedPlan(FusedOperation fused) {
        if (fused == null || fused.getPlan() == null || fused.getPrecisionMode() != FusedDTypeOps.MODE_BF16) {
            return false;
        }
        int directStridedInputs = 0;
        int directContiguousInputs = 0;
        for (FusedExternalInputPlan input : fused.getPlan().inputs()) {
            if (input.dataType() != DataType.BFLOAT16) {
                return false;
            }
            switch (input.accessKind()) {
                case DIRECT_STRIDED -> directStridedInputs++;
                case DIRECT_CONTIGUOUS -> directContiguousInputs++;
                default -> {
                    return false;
                }
            }
        }
        if (directStridedInputs != 1 || directContiguousInputs < 3) {
            return false;
        }

        boolean hasDivision = false;
        for (FusedNodePlan node : fused.getPlan().nodes()) {
            if (node.outputType() != DataType.BFLOAT16) {
                return false;
            }
            switch (node.opType()) {
                case NEG, ADD, SUB, MUL, DIV, INV, MUL_SCALAR -> {
                    if (node.opType() == Operation.OpType.DIV || node.opType() == Operation.OpType.INV) {
                        hasDivision = true;
                    }
                }
                default -> {
                    return false;
                }
            }
        }
        return hasDivision && fused.getPlan().nodeCount() >= 5;
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
