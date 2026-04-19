package backend.kernels.cpu.elementwise.strided;

import backend.kernels.cpu.layout.BroadcastPlanResolver;
import backend.kernels.cpu.plan.CpuExecutionPlanner;
import operations.Operation;
import tensor.DataType;
import tensor.Tensor;

import java.util.Arrays;
import java.util.List;

public final class StridedPathEligibility {
    private StridedPathEligibility() {
    }

    public static boolean canUse(
            Operation op,
            List<Tensor> inputs,
            Tensor node,
            DataType targetType,
            CpuExecutionPlanner planner
    ) {
        if (op == null || node == null || inputs.isEmpty()) {
            return false;
        }
        if (op.opType() == Operation.OpType.CONTIGUOUS) {
            return false;
        }
        if (op.opType().category() != Operation.OpArityClass.ELEMENT_WISE || !CpuStridedElementWise.supports(op)) {
            return false;
        }
        if (BroadcastPlanResolver.requiresBinaryBroadcast(op, inputs, node)) {
            return false;
        }

        boolean hasOffsetInput = false;
        boolean hasNonContiguousInput = false;
        int[] outShape = node.getShapeUnsafe();

        for (int i = 0; i < inputs.size(); i++) {
            Tensor input = inputs.get(i);
            if (input == null) {
                return false;
            }
            if (!isInputTypeCompatible(op, input, targetType, i)) {
                return false;
            }
            if (!Arrays.equals(input.getShapeUnsafe(), outShape)) {
                return false;
            }
            if (input.hasStorageOffset()) {
                hasOffsetInput = true;
            }
            if (!input.isContiguous()) {
                hasNonContiguousInput = true;
            }
        }

        if (!hasOffsetInput && !hasNonContiguousInput) {
            return false;
        }

        if (hasOffsetInput && !hasNonContiguousInput) {
            return true;
        }

        return !planner.shouldMaterializeNonContiguous(node.getFlatDataSize());
    }

    private static boolean isInputTypeCompatible(Operation op, Tensor input, DataType targetType, int inputIndex) {
        if (op == null || input == null) {
            return false;
        }
        return switch (op.opType()) {
            case GT, GE, LT, LE, EQ, NE ->
                    input.getDataType() == DataType.FLOAT64
                            || input.getDataType() == DataType.FLOAT32
                            || input.getDataType() == DataType.BFLOAT16;
            case WHERE -> inputIndex == 0
                    ? input.getDataType() == DataType.BOOL
                    : input.getDataType() == targetType;
            case LOGICAL_AND, LOGICAL_OR, LOGICAL_NOT -> input.getDataType() == DataType.BOOL;
            default -> input.getDataType() == targetType;
        };
    }
}
