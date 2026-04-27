package backend.cpu.kernels.elementwise.strided;

import backend.cpu.kernels.layout.BroadcastPlanResolver;
import backend.cpu.kernels.plan.CpuExecutionPlanner;
import operations.Operation;
import tensor.DataType;
import tensor.Tensor;

import java.util.Arrays;
import java.util.List;

public final class StridedPathEligibility {
    private enum NonContiguousClass {
        RANK2_TRANSPOSE_LIKE,
        RANK2_INNER_STRIDED,
        RANK2_OUTER_STRIDED,
        RANK2_ROW_BROADCAST,
        RANK2_COL_BROADCAST,
        OTHER;

        boolean cheapMaterializationFriendly() {
            return this == RANK2_TRANSPOSE_LIKE || this == RANK2_INNER_STRIDED;
        }
    }

    private StridedPathEligibility() {
    }

    public static StridedLayoutDecision resolve(
            Operation op,
            List<Tensor> inputs,
            Tensor node,
            DataType targetType,
            CpuExecutionPlanner planner
    ) {
        if (op == null || node == null || inputs.isEmpty()) {
            return StridedLayoutDecision.NONE;
        }
        if (op.opType() == Operation.OpType.CONTIGUOUS) {
            return StridedLayoutDecision.NONE;
        }
        if (op.opType().category() != Operation.OpArityClass.ELEMENT_WISE || !CpuStridedElementWise.supports(op)) {
            return StridedLayoutDecision.NONE;
        }
        if (BroadcastPlanResolver.requiresBinaryBroadcast(op, inputs, node)) {
            return StridedLayoutDecision.NONE;
        }

        boolean hasOffsetInput = false;
        boolean hasNonContiguousInput = false;
        int nonContiguousInputCount = 0;
        int firstNonContiguousInput = -1;
        boolean allNonContiguousCheapMaterializationFriendly = true;
        int[] outShape = node.getShapeUnsafe();

        for (int i = 0; i < inputs.size(); i++) {
            Tensor input = inputs.get(i);
            if (input == null) {
                return StridedLayoutDecision.NONE;
            }
            if (!isInputTypeCompatible(op, input, targetType, i)) {
                return StridedLayoutDecision.NONE;
            }
            if (!Arrays.equals(input.getShapeUnsafe(), outShape)) {
                return StridedLayoutDecision.NONE;
            }
            if (input.hasStorageOffset()) {
                hasOffsetInput = true;
            }
            if (!input.isContiguous()) {
                hasNonContiguousInput = true;
                nonContiguousInputCount++;
                if (firstNonContiguousInput < 0) {
                    firstNonContiguousInput = i;
                }
                NonContiguousClass nonContiguousClass = classifyNonContiguous(input);
                if (!nonContiguousClass.cheapMaterializationFriendly()) {
                    allNonContiguousCheapMaterializationFriendly = false;
                }
            }
        }

        if (!hasOffsetInput && !hasNonContiguousInput) {
            return StridedLayoutDecision.NONE;
        }

        if (hasOffsetInput && !hasNonContiguousInput) {
            return StridedLayoutDecision.KEEP_STRIDED;
        }

        boolean preferMaterialize = allNonContiguousCheapMaterializationFriendly
                && planner.shouldMaterializeCheapStridedElementwise(op, targetType, node.getFlatDataSize());
        if (!preferMaterialize && !planner.shouldMaterializeNonContiguous(node.getFlatDataSize())) {
            return StridedLayoutDecision.KEEP_STRIDED;
        }

        if (!hasOffsetInput && inputs.size() <= 2 && nonContiguousInputCount == 1) {
            return firstNonContiguousInput == 0
                    ? StridedLayoutDecision.MATERIALIZE_INPUT_0
                    : StridedLayoutDecision.MATERIALIZE_INPUT_1;
        }

        return StridedLayoutDecision.MATERIALIZE_ALL;
    }

    private static NonContiguousClass classifyNonContiguous(Tensor input) {
        int[] shape = input.getShapeUnsafe();
        int[] strides = input.getStridesUnsafe();
        if (shape != null && strides != null && shape.length == 2 && strides.length == 2) {
            if (isRank2TransposeLike(shape, strides)) {
                return NonContiguousClass.RANK2_TRANSPOSE_LIKE;
            }
            if (isRank2RowBroadcast(strides)) {
                return NonContiguousClass.RANK2_ROW_BROADCAST;
            }
            if (isRank2ColBroadcast(strides)) {
                return NonContiguousClass.RANK2_COL_BROADCAST;
            }
            if (isRank2InnerStrided(strides)) {
                return NonContiguousClass.RANK2_INNER_STRIDED;
            }
            if (isRank2OuterStrided(strides)) {
                return NonContiguousClass.RANK2_OUTER_STRIDED;
            }
        }
        return NonContiguousClass.OTHER;
    }

    private static boolean isRank2TransposeLike(int[] shape, int[] strides) {
        if (shape == null || strides == null || shape.length != 2 || strides.length != 2) {
            return false;
        }
        return strides[0] == 1 && strides[1] >= Math.max(1, shape[0]);
    }

    private static boolean isRank2RowBroadcast(int[] strides) {
        return strides != null && strides.length == 2 && strides[0] > 0 && strides[1] == 0;
    }

    private static boolean isRank2ColBroadcast(int[] strides) {
        return strides != null && strides.length == 2 && strides[0] == 0 && strides[1] > 0;
    }

    private static boolean isRank2InnerStrided(int[] strides) {
        return strides != null && strides.length == 2 && strides[1] > 1;
    }

    private static boolean isRank2OuterStrided(int[] strides) {
        return strides != null && strides.length == 2 && strides[0] > 1 && strides[1] == 1;
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
