package backend.cpu.prepare.layout;

import backend.cpu.prepare.CpuExecutionPlanner;
import backend.cpu.plan.PreparedTypeContract;
import operations.Operation;
import tensor.DataType;
import graph.compile.descriptor.CompiledTensorDescriptor;

import java.util.List;

final class PreparedInputPolicy {
    private PreparedInputPolicy() {
    }

    static boolean bypassPreparation(Operation op) {
        if (op == null || op.opType() == null) {
            return false;
        }
        return switch (op.opType()) {
            case CONTIGUOUS, RESHAPE, EXPAND, SELECT, SLICE, SLICE_SCATTER_ADD, CONCAT, PAD, TILE, UNFOLD_AXIS, PERMUTE, EXPAND_DIMS, SQUEEZE, CAST,
                    GATHER_AXIS, GATHER_ND,
                    SUM, MEAN, REDUCE_MIN, REDUCE_MAX, REDUCE_PROD, CUMSUM, ARGMAX,
                    SOFTMAX, LOG_SOFTMAX, SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS,
                    NLL_LOSS, CROSS_ENTROPY_LOSS, CROSS_ENTROPY_LOSS_INDICES, NOOP -> true;
            default -> false;
        };
    }

    static boolean requiresPreparedInputs(
            Operation op,
            List<CompiledTensorDescriptor> inputs,
            CompiledTensorDescriptor node,
            PreparedTypeContract typeContract,
            CpuExecutionPlanner planner
    ) {
        if (inputs == null || inputs.isEmpty()) {
            return false;
        }
        for (int i = 0; i < inputs.size(); i++) {
            CompiledTensorDescriptor input = inputs.get(i);
            if (input == null) {
                return true;
            }
            DataType expectedInputType = typeContract.expectedInputTypes().get(i);
            if (requiresPreparedInput(op, input, node, expectedInputType, planner)) {
                return true;
            }
        }
        return false;
    }

    static boolean requiresPreparedInput(
            Operation op,
            CompiledTensorDescriptor input,
            CompiledTensorDescriptor node,
            DataType expectedInputType,
            CpuExecutionPlanner planner
    ) {
        if (input.dataType() != expectedInputType) {
            return true;
        }

        if (input.hasStorageOffset()) {
            if (op == null) {
                return true;
            }
            return switch (op.opType()) {
                case FUSED -> false;
                case RESHAPE, EXPAND, SELECT, SLICE, SLICE_SCATTER_ADD, CONCAT, PAD, TILE, UNFOLD_AXIS, PERMUTE, EXPAND_DIMS, SQUEEZE, CAST,
                        GATHER, GATHER_AXIS,
                        GATHER_ND, TAKE_ALONG_AXIS, SCATTER_ADD, SCATTER_AXIS_ADD, SCATTER_ELEMENTS, SCATTER_ND,
                        SUM, MEAN, REDUCE_MIN, REDUCE_MAX, REDUCE_PROD, CUMSUM, ARGMAX, REDUCE_ALL, REDUCE_ANY,
                        SOFTMAX, LOG_SOFTMAX, SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS,
                        NLL_LOSS, CROSS_ENTROPY_LOSS, CROSS_ENTROPY_LOSS_INDICES,
                        NOOP -> false;
                default -> true;
            };
        }

        if (input.contiguous()) {
            return false;
        }

        if (op == null) {
            return false;
        }

        return switch (op.opType()) {
            case FUSED -> false;
            case CONTIGUOUS, RESHAPE, EXPAND, SELECT, SLICE, SLICE_SCATTER_ADD, CONCAT, PAD, TILE, UNFOLD_AXIS, PERMUTE, EXPAND_DIMS, SQUEEZE, CAST,
                    GATHER_AXIS, GATHER_ND,
                    SUM, MEAN, REDUCE_MIN, REDUCE_MAX, REDUCE_PROD, CUMSUM, ARGMAX, REDUCE_ALL, REDUCE_ANY,
                    SOFTMAX, LOG_SOFTMAX, SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS,
                    NLL_LOSS, CROSS_ENTROPY_LOSS, CROSS_ENTROPY_LOSS_INDICES,
                    NOOP -> false;
            case LAYER_NORM, RMS_NORM, SCALED_DOT_PRODUCT_ATTENTION -> true;
            case MATMUL, LINEAR, CONV2D, MAX_POOL2D, AVG_POOL2D -> true;
            case GT, GE, LT, LE, EQ, NE, WHERE, LOGICAL_AND, LOGICAL_OR, LOGICAL_NOT -> !input.contiguous();
            default -> op.opType().category() == Operation.OpArityClass.ELEMENT_WISE
                    && planner.shouldMaterializeNonContiguous(Math.toIntExact(node.logicalElementCount()));
        };
    }

    static boolean canConvertPreparedInput(DataType sourceType, DataType expectedType) {
        if (sourceType == expectedType) {
            return true;
        }
        if (sourceType == DataType.BOOL || expectedType == DataType.BOOL || sourceType == DataType.INT32 || expectedType == DataType.INT32) {
            return false;
        }
        return true;
    }
}
