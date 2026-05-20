package backend.cpu.kernels.elementwise.strided;

import backend.cpu.kernels.layout.BroadcastPlanResolver;
import backend.cpu.kernels.plan.CpuExecutionPlanner;
import operations.Operation;
import tensor.DataType;
import graph.compile.descriptor.CompiledTensorDescriptor;

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
            List<CompiledTensorDescriptor> inputs,
            CompiledTensorDescriptor node,
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
        int[] outShape = node.shape();

        for (int i = 0; i < inputs.size(); i++) {
            CompiledTensorDescriptor input = inputs.get(i);
            if (input == null) {
                return StridedLayoutDecision.NONE;
            }
            if (!isInputTypeCompatible(op, input, targetType, i)) {
                return StridedLayoutDecision.NONE;
            }
            if (!Arrays.equals(input.shape(), outShape)) {
                return StridedLayoutDecision.NONE;
            }
            if (input.hasStorageOffset()) {
                hasOffsetInput = true;
            }
            if (!input.contiguous()) {
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
                && planner.shouldMaterializeCheapStridedElementwise(op, targetType, Math.toIntExact(node.logicalElementCount()));
        if (!preferMaterialize && !planner.shouldMaterializeNonContiguous(Math.toIntExact(node.logicalElementCount()))) {
            return StridedLayoutDecision.KEEP_STRIDED;
        }

        if (!hasOffsetInput && inputs.size() <= 2 && nonContiguousInputCount == 1) {
            return firstNonContiguousInput == 0
                    ? StridedLayoutDecision.MATERIALIZE_INPUT_0
                    : StridedLayoutDecision.MATERIALIZE_INPUT_1;
        }

        return StridedLayoutDecision.MATERIALIZE_ALL;
    }

    private static NonContiguousClass classifyNonContiguous(CompiledTensorDescriptor input) {
        int[] shape = input.shape();
        int[] strides = input.strides();
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

    private static boolean isInputTypeCompatible(Operation op, CompiledTensorDescriptor input, DataType targetType, int inputIndex) {
        if (op == null || input == null) {
            return false;
        }
        return switch (op.opType()) {
            case GT, GE, LT, LE, EQ, NE ->
                    input.dataType() == DataType.FLOAT64
                            || input.dataType() == DataType.FLOAT32
                            || input.dataType() == DataType.BFLOAT16;
            case WHERE -> inputIndex == 0
                    ? input.dataType() == DataType.BOOL
                    : input.dataType() == targetType;
            case LOGICAL_AND, LOGICAL_OR, LOGICAL_NOT -> input.dataType() == DataType.BOOL;
            default -> input.dataType() == targetType;
        };
    }
}
