package backend.cpu.fused.plan;

import backend.cpu.fused.ir.FusedAccessKind;
import backend.cpu.fused.ir.FusedExpressionPlan;
import backend.cpu.fused.ir.FusedExternalInputPlan;
import backend.cpu.fused.ir.FusedNodePlan;
import backend.cpu.fused.ir.ScalarDoubleAttribute;
import backend.cpu.kernels.CpuComputeDType;
import backend.cpu.kernels.ResolvedCpuComputeContract;
import backend.cpu.fused.numeric.FusedComputeKind;
import backend.cpu.fused.numeric.FusedNumericContract;
import backend.cpu.fused.numeric.FusedStorageKind;
import backend.cpu.fused.numeric.FusedValueLane;
import operations.Operation;
import tensor.DataType;

/**
 * Recognizes fused expression shapes with measured scalar/vector constraints.
 */
public final class FusedVectorGuard {
    private FusedVectorGuard() {
    }

    public static boolean requiresScalarDispatch(FusedOperation fused) {
        return dispatchBlockReason(fused).forceSerialScalarDispatch();
    }

    public static boolean requiresScalarAsmWidth(ResolvedCpuComputeContract contract, FusedOperation fused) {
        return asmWidthBlockReason(contract, fused).forceScalarAsmWidth();
    }

    public static FusedVectorBlockReason dispatchBlockReason(FusedOperation fused) {
        return isMaskedScaleWherePlan(fused)
                ? FusedVectorBlockReason.MASKED_SCALE_WHERE_SCALAR_ONLY
                : FusedVectorBlockReason.NONE;
    }

    public static FusedVectorBlockReason asmWidthBlockReason(
            ResolvedCpuComputeContract contract,
            FusedOperation fused
    ) {
        FusedVectorBlockReason dispatchReason = dispatchBlockReason(fused);
        if (dispatchReason.forceScalarAsmWidth()) {
            return dispatchReason;
        }
        if (fused != null && !supportsAllocationFreeVectorPath(fused.getNumericContract(), fused.getPlan())) {
            return FusedVectorBlockReason.UNSUPPORTED_ALLOCATION_FREE_VECTOR_PATH;
        }
        if (fused != null
                && fused.getNumericContract().usesMemorySegmentStorage()
                && !supportsMemorySegmentVectorAsm(fused.getNumericContract(), fused.getPlan())) {
            return FusedVectorBlockReason.MEMORY_SEGMENT_SCALAR_ONLY;
        }
        return isBf16AffineRationalStridedPlan(contract, fused)
                ? FusedVectorBlockReason.BF16_STRIDED_RATIONAL_SCALAR_ONLY
                : FusedVectorBlockReason.NONE;
    }

    public static FusedVectorBlockReason preparedBlockReason(
            ResolvedCpuComputeContract contract,
            FusedOperation fused,
            int totalLength,
            int cpuVectorMinSize,
            int asmVectorWidth
    ) {
        FusedVectorBlockReason explicitReason = asmWidthBlockReason(contract, fused);
        if (explicitReason != FusedVectorBlockReason.NONE) {
            return explicitReason;
        }
        if (asmVectorWidth <= 1) {
            return FusedVectorBlockReason.PREFERRED_WIDTH_SCALAR;
        }
        return totalLength < Math.max(1, cpuVectorMinSize)
                ? FusedVectorBlockReason.BELOW_VECTOR_THRESHOLD
                : FusedVectorBlockReason.NONE;
    }

    public static boolean supportsMemorySegmentVectorAsm(
            FusedNumericContract numericContract,
            FusedExpressionPlan plan
    ) {
        if (numericContract == null || plan == null) {
            return false;
        }
        if (numericContract.inputStorageKind() != FusedStorageKind.CPU_MEMORY_SEGMENT
                || numericContract.outputStorageKind() != FusedStorageKind.CPU_MEMORY_SEGMENT) {
            return false;
        }
        if (!usesAllocationFreeSegmentVectorLanes(numericContract)) {
            return false;
        }
        DataType outputType = plan.outputNode().outputType();
        if (!isAllocationFreeSegmentVectorOutput(outputType)) {
            return false;
        }
        for (int i = 0; i < plan.inputCount(); i++) {
            FusedExternalInputPlan input = plan.inputs().get(i);
            if (!isF32OrF64(input.dataType())
                    || !matchesSegmentVectorLane(input.dataType(), numericContract)
                    || input.dataType() == DataType.BOOL) {
                return false;
            }
        }
        for (FusedNodePlan node : plan.nodes()) {
            if (!supportsDirectVectorNode(node)) {
                return false;
            }
        }
        return true;
    }

    public static boolean supportsAllocationFreeVectorPath(
            FusedNumericContract numericContract,
            FusedExpressionPlan plan
    ) {
        if (numericContract == null || plan == null) {
            return false;
        }
        if (numericContract.writesBf16()) {
            return false;
        }
        if (numericContract.computeKind() != FusedComputeKind.F32
                && numericContract.computeKind() != FusedComputeKind.F64) {
            return false;
        }
        if (!isF32OrF64(plan.outputNode().outputType())) {
            return false;
        }
        for (FusedExternalInputPlan input : plan.inputs()) {
            if (!supportsGeneratedVectorInput(input, numericContract)) {
                return false;
            }
        }
        for (FusedNodePlan node : plan.nodes()) {
            if (!supportsDirectVectorNode(node)) {
                return false;
            }
        }
        return true;
    }

    public static boolean shouldUseConservativeVectorThreshold(FusedOperation fused) {
        if (fused == null || fused.getPlan() == null) {
            return false;
        }
        if ((fused.getDispatchFamily() == FusedDispatchFamily.CHEAP_CONTIGUOUS
                || fused.getDispatchFamily() == FusedDispatchFamily.CHEAP_STRIDED)
                && fused.getPlan().nodeCount() <= 2) {
            return true;
        }
        return fused.getDispatchFamily() == FusedDispatchFamily.NON_CHEAP_STRIDED
                && !isVectorFriendlyNonCheapStridedPlan(fused);
    }

    private static boolean isMaskedScaleWherePlan(FusedOperation fused) {
        if (fused == null || fused.getPlan() == null) {
            return false;
        }
        var plan = fused.getPlan();
        FusedNumericContract numericContract = fused.getNumericContract();
        if (!numericContract.usesFloatCompute() || numericContract.writesBf16()
                || plan.inputCount() != 3 || plan.nodeCount() != 2) {
            return false;
        }
        FusedExternalInputPlan maskInput = plan.inputs().get(0);
        FusedExternalInputPlan fillInput = plan.inputs().get(1);
        FusedExternalInputPlan valueInput = plan.inputs().get(2);
        if (maskInput.dataType() != DataType.BOOL
                || fillInput.dataType() != DataType.FLOAT32
                || valueInput.dataType() != DataType.FLOAT32
                || !isContiguousLinear(maskInput)
                || !isContiguousLinear(valueInput)
                || fillInput.accessKind() != FusedAccessKind.BROADCAST_STRIDED
                || !isZeroStrideBroadcast(fillInput)) {
            return false;
        }
        FusedNodePlan scaleNode = plan.nodes().get(0);
        FusedNodePlan whereNode = plan.nodes().get(1);
        if (scaleNode.opType() != Operation.OpType.MUL_SCALAR
                || whereNode.opType() != Operation.OpType.WHERE
                || scaleNode.outputType() != DataType.FLOAT32
                || whereNode.outputType() != DataType.FLOAT32
                || !(scaleNode.attributes() instanceof ScalarDoubleAttribute)
                || scaleNode.inputRefs().size() != 1
                || scaleNode.inputRefs().get(0) != 2
                || whereNode.inputRefs().size() != 3
                || whereNode.inputRefs().get(0) != 0
                || plan.outputNode().index() != whereNode.index()) {
            return false;
        }
        int scaledValueRef = plan.inputCount() + scaleNode.index();
        return (whereNode.inputRefs().get(1) == 1 && whereNode.inputRefs().get(2) == scaledValueRef)
                || (whereNode.inputRefs().get(1) == scaledValueRef && whereNode.inputRefs().get(2) == 1);
    }

    private static boolean isBf16AffineRationalStridedPlan(
            ResolvedCpuComputeContract contract,
            FusedOperation fused
    ) {
        if (contract == null
                || contract.computeType() != CpuComputeDType.BF16_NATIVE
                || fused == null
                || fused.getPlan() == null
                || fused.getDispatchFamily() != FusedDispatchFamily.NON_CHEAP_STRIDED
                || !fused.getNumericContract().writesBf16()) {
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

    private static boolean isVectorFriendlyNonCheapStridedPlan(FusedOperation fused) {
        if (fused == null || fused.getPlan() == null) {
            return false;
        }
        if (containsTranscendental(fused)) {
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

    private static boolean containsTranscendental(FusedOperation fused) {
        if (fused == null || fused.getPlan() == null) {
            return false;
        }
        return fused.getPlan().nodes().stream().anyMatch(node -> switch (node.opType()) {
            case EXP, FAST_EXP, TANH, FAST_TANH, LOG, SIGMOID, POW -> true;
            default -> false;
        });
    }

    private static boolean usesAllocationFreeSegmentVectorLanes(FusedNumericContract numericContract) {
        // MemorySegment vector ASM intentionally admits only F32/F64 lanes the emitter can
        // load, compute, and store without runtime helper loops. General segment gather
        // uses generated lane loads into method-local scratch allocated outside the loop.
        // BF16 and BOOL/mask segment paths stay scalar until they have dedicated emitters.
        if (numericContract.computeKind() != FusedComputeKind.F32
                && numericContract.computeKind() != FusedComputeKind.F64) {
            return false;
        }
        return numericContract.outputValueLane() == FusedValueLane.F32
                || numericContract.outputValueLane() == FusedValueLane.F64;
    }

    private static boolean isAllocationFreeSegmentVectorOutput(DataType dataType) {
        return dataType == DataType.FLOAT32 || dataType == DataType.FLOAT64;
    }

    private static boolean isF32OrF64(DataType dataType) {
        return dataType == DataType.FLOAT32 || dataType == DataType.FLOAT64;
    }

    private static boolean isAllocationFreeNumericVectorOp(Operation.OpType opType) {
        if (opType == null) {
            return false;
        }
        return switch (opType) {
            case ADD, SUB, MUL, DIV, MIN, MAX,
                    NEG, INV, ABS, SQRT,
                    CONST_SCALAR, MUL_SCALAR,
                    RELU, CLAMP_MIN, CLAMP_MAX,
                    NOOP,
                    GT, GE, LT, LE, EQ, NE,
                    LOGICAL_AND, LOGICAL_OR, LOGICAL_NOT,
                    WHERE -> true;
            case POW -> false;
            default -> false;
        };
    }

    private static boolean supportsGeneratedVectorInput(
            FusedExternalInputPlan input,
            FusedNumericContract numericContract
    ) {
        if (input == null || input.dataType() == DataType.BOOL) {
            return false;
        }
        if (!isF32OrF64(input.dataType()) || !matchesSegmentVectorLane(input.dataType(), numericContract)) {
            return false;
        }
        return true;
    }

    private static boolean supportsDirectVectorNode(FusedNodePlan node) {
        if (node == null) {
            return false;
        }
        if (node.opType() == Operation.OpType.POW) {
            return isDirectVectorPowSpecialCase(node);
        }
        return isAllocationFreeNumericVectorOp(node.opType()) && supportsNodeOutputType(node);
    }

    private static boolean supportsNodeOutputType(FusedNodePlan node) {
        return node.outputType() == DataType.BOOL || isF32OrF64(node.outputType());
    }

    private static boolean isDirectVectorPowSpecialCase(FusedNodePlan node) {
        if (!(node.attributes() instanceof ScalarDoubleAttribute attribute)) {
            return false;
        }
        double exponent = attribute.value();
        return Double.compare(exponent, -2.0d) == 0
                || Double.compare(exponent, -1.0d) == 0
                || Double.compare(exponent, 0.0d) == 0
                || Double.compare(exponent, 1.0d) == 0
                || Double.compare(exponent, 2.0d) == 0
                || Double.compare(exponent, 0.5d) == 0;
    }

    private static boolean isContiguousLinear(FusedExternalInputPlan input) {
        return input.accessKind() == FusedAccessKind.DIRECT_CONTIGUOUS
                || input.accessKind() == FusedAccessKind.OFFSET_CONTIGUOUS;
    }

    private static boolean matchesSegmentVectorLane(
            DataType dataType,
            FusedNumericContract numericContract
    ) {
        if (numericContract.computeKind() == FusedComputeKind.F32) {
            return dataType == DataType.FLOAT32;
        }
        if (numericContract.computeKind() == FusedComputeKind.F64) {
            return dataType == DataType.FLOAT64;
        }
        return false;
    }

    private static boolean isZeroStrideBroadcast(FusedExternalInputPlan input) {
        for (int stride : input.effectiveStrides()) {
            if (stride != 0) {
                return false;
            }
        }
        return true;
    }
}
