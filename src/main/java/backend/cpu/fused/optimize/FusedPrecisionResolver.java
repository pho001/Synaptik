package backend.cpu.fused.optimize;

import backend.cpu.fused.codegen.FusedDTypeOps;
import backend.cpu.fused.codegen.FusedExpressionPlan;
import backend.cpu.fused.codegen.FusedExternalInputPlan;
import backend.cpu.fused.codegen.FusedNodePlan;
import tensor.DataType;

/**
 * Internal precision resolver for fused CPU expression generation.
 */
public final class FusedPrecisionResolver {
    public static int resolve(FusedExpressionPlan plan) {
        DataType target = null;
        if (plan != null) {
            for (FusedExternalInputPlan input : plan.inputs()) {
                target = resolveTarget(target, input.dataType());
            }
            for (FusedNodePlan node : plan.nodes()) {
                target = resolveTarget(target, node.outputType());
            }
        }
        return switch (target) {
            case null -> FusedDTypeOps.MODE_F32;
            case FLOAT64 -> FusedDTypeOps.MODE_F64;
            case FLOAT32 -> FusedDTypeOps.MODE_F32;
            case BFLOAT16 -> FusedDTypeOps.MODE_BF16;
            case INT32, INT64, BOOL -> throw new UnsupportedOperationException("INT32/INT64/BOOL tensors are not supported in fused precision resolution.");
        };
    }

    private static DataType resolveTarget(DataType current, DataType next) {
        if (next == null || next == DataType.BOOL) {
            return current;
        }
        if (next == DataType.INT32 || next == DataType.INT64) {
            throw new UnsupportedOperationException("INT32/INT64 tensors are not supported in fused precision resolution.");
        }
        if (next == DataType.FLOAT64) {
            return DataType.FLOAT64;
        }
        if (next == DataType.FLOAT32) {
            return current == DataType.FLOAT64 ? current : DataType.FLOAT32;
        }
        if (next == DataType.BFLOAT16 && current == null) {
            return DataType.BFLOAT16;
        }
        return current;
    }
}
