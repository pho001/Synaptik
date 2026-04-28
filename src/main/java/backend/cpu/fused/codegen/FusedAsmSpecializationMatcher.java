package backend.cpu.fused.codegen;

import operations.Operation;
import tensor.DataType;

/**
 * Internal matcher that selects specialized fused ASM generation paths.
 */
public final class FusedAsmSpecializationMatcher {
    private FusedAsmSpecializationMatcher() {}

    public static FusedAsmSpecializationKind match(FusedExpressionPlan plan, int precisionMode) {
        if (plan == null || precisionMode != FusedDTypeOps.MODE_F32) {
            return FusedAsmSpecializationKind.NONE;
        }
        if (plan.inputCount() != 3 || plan.nodeCount() != 2) {
            return FusedAsmSpecializationKind.NONE;
        }

        FusedExternalInputPlan maskInput = plan.inputs().get(0);
        FusedExternalInputPlan fillInput = plan.inputs().get(1);
        FusedExternalInputPlan valueInput = plan.inputs().get(2);
        if (maskInput.dataType() != DataType.BOOL
                || fillInput.dataType() != DataType.FLOAT32
                || valueInput.dataType() != DataType.FLOAT32) {
            return FusedAsmSpecializationKind.NONE;
        }
        if (!isContiguousLinear(maskInput) || !isContiguousLinear(valueInput)) {
            return FusedAsmSpecializationKind.NONE;
        }
        if (fillInput.accessKind() != FusedAccessKind.BROADCAST_STRIDED || !isZeroStrideBroadcast(fillInput)) {
            return FusedAsmSpecializationKind.NONE;
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
            return FusedAsmSpecializationKind.NONE;
        }

        int scaledValueRef = plan.inputCount() + scaleNode.index();
        if (whereNode.inputRefs().get(1) == 1 && whereNode.inputRefs().get(2) == scaledValueRef) {
            return FusedAsmSpecializationKind.F32_MASKED_SCALE_WHERE;
        }
        if (whereNode.inputRefs().get(1) == scaledValueRef && whereNode.inputRefs().get(2) == 1) {
            return FusedAsmSpecializationKind.F32_MASKED_SCALE_WHERE_INVERTED;
        }
        return FusedAsmSpecializationKind.NONE;
    }

    public static float requireF32MaskedScaleWhereScalar(FusedExpressionPlan plan) {
        FusedAsmSpecializationKind kind = match(plan, FusedDTypeOps.MODE_F32);
        if (kind != FusedAsmSpecializationKind.F32_MASKED_SCALE_WHERE
                && kind != FusedAsmSpecializationKind.F32_MASKED_SCALE_WHERE_INVERTED) {
            throw new IllegalArgumentException("Plan does not match F32 masked-scale-where specialization.");
        }
        return (float) ((ScalarDoubleAttribute) plan.nodes().get(0).attributes()).value();
    }

    private static boolean isContiguousLinear(FusedExternalInputPlan input) {
        return input.accessKind() == FusedAccessKind.DIRECT_CONTIGUOUS
                || input.accessKind() == FusedAccessKind.OFFSET_CONTIGUOUS;
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
