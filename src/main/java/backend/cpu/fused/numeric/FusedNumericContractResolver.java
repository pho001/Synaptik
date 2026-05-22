package backend.cpu.fused.numeric;

import backend.cpu.fused.ir.FusedExpressionPlan;
import backend.cpu.fused.ir.FusedExternalInputPlan;
import backend.cpu.fused.ir.FusedNodePlan;
import tensor.DataType;

/**
 * Resolves fused storage and compute numeric facts before execution preparation.
 */
public final class FusedNumericContractResolver {
    private FusedNumericContractResolver() {
    }

    public static FusedNumericContract resolve(FusedExpressionPlan plan) {
        return resolve(plan, FusedStorageKind.CPU_JAVA_ARRAY, FusedStorageKind.CPU_JAVA_ARRAY);
    }

    public static FusedNumericContract resolve(
            FusedExpressionPlan plan,
            FusedStorageKind inputStorageKind,
            FusedStorageKind outputStorageKind
    ) {
        DataType target = null;
        if (plan != null) {
            for (FusedExternalInputPlan input : plan.inputs()) {
                target = resolveTarget(target, input.dataType());
            }
            for (FusedNodePlan node : plan.nodes()) {
                target = resolveTarget(target, node.outputType());
            }
        }
        DataType storageType = target == null ? DataType.FLOAT32 : target;
        FusedValueLane lane = FusedValueLane.fromDataType(storageType);
        FusedComputeKind computeKind = storageType == DataType.FLOAT64
                ? FusedComputeKind.F64
                : FusedComputeKind.F32;
        return new FusedNumericContract(
                inputStorageKind,
                outputStorageKind,
                lane,
                computeKind,
                lane
        );
    }

    private static DataType resolveTarget(DataType current, DataType next) {
        if (next == null || next == DataType.BOOL) {
            return current;
        }
        if (next == DataType.INT32 || next == DataType.INT64) {
            throw new UnsupportedOperationException("INT32/INT64 tensors are not supported in fused numeric contract resolution.");
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
