package backend.cpu.kernels.layout;

import backend.cpu.kernels.plan.CpuExecutionPlanner;
import backend.cpu.kernels.plan.PreparedTypeContract;
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
            case CONTIGUOUS, RESHAPE, EXPAND, SELECT, SLICE, SLICE_GRAD, SLICE_SCATTER_ADD, CONCAT, PAD, TILE, PERMUTE, EXPAND_DIMS, SQUEEZE, CAST,
                    GATHER_AXIS, GATHER_AXIS_GRAD, GATHER_ND, GATHER_ND_GRAD,
                    SUM, MEAN, REDUCE_MIN, REDUCE_MAX, REDUCE_PROD, CUMSUM, ARGMAX,
                    SOFTMAX, SOFTMAX_GRAD, LOG_SOFTMAX, LOG_SOFTMAX_GRAD, SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS,
                    NLL_LOSS, CROSS_ENTROPY_LOSS, CROSS_ENTROPY_LOSS_INDICES, CROSS_ENTROPY_LOSS_INDICES_GRAD,
                    REDUCE_MIN_GRAD, REDUCE_MAX_GRAD, NOOP -> true;
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
                case RESHAPE, EXPAND, SELECT, SLICE, SLICE_GRAD, SLICE_SCATTER_ADD, CONCAT, PAD, TILE, PERMUTE, EXPAND_DIMS, SQUEEZE, CAST,
                        GATHER, GATHER_GRAD, GATHER_AXIS, GATHER_AXIS_GRAD,
                        GATHER_ND, GATHER_ND_GRAD, TAKE_ALONG_AXIS, TAKE_ALONG_AXIS_GRAD, SCATTER_ADD, SCATTER_AXIS_ADD, SCATTER_ELEMENTS, SCATTER_ND,
                        SUM, MEAN, REDUCE_MIN, REDUCE_MAX, REDUCE_PROD, CUMSUM, ARGMAX, REDUCE_ALL, REDUCE_ANY,
                        SOFTMAX, SOFTMAX_GRAD, LOG_SOFTMAX, LOG_SOFTMAX_GRAD, SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS,
                        NLL_LOSS, CROSS_ENTROPY_LOSS, CROSS_ENTROPY_LOSS_INDICES, CROSS_ENTROPY_LOSS_INDICES_GRAD,
                        MIN_GRAD, MAX_GRAD, REDUCE_MIN_GRAD, REDUCE_MAX_GRAD,
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
            case CONTIGUOUS, RESHAPE, EXPAND, SELECT, SLICE, SLICE_GRAD, SLICE_SCATTER_ADD, CONCAT, PAD, TILE, PERMUTE, EXPAND_DIMS, SQUEEZE, CAST,
                    GATHER_AXIS, GATHER_AXIS_GRAD, GATHER_ND, GATHER_ND_GRAD,
                    SUM, MEAN, REDUCE_MIN, REDUCE_MAX, REDUCE_PROD, CUMSUM, ARGMAX, REDUCE_ALL, REDUCE_ANY,
                    SOFTMAX, SOFTMAX_GRAD, LOG_SOFTMAX, LOG_SOFTMAX_GRAD, SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS,
                    NLL_LOSS, CROSS_ENTROPY_LOSS, CROSS_ENTROPY_LOSS_INDICES, CROSS_ENTROPY_LOSS_INDICES_GRAD,
                    NOOP -> false;
            case LAYER_NORM, RMS_NORM, SCALED_DOT_PRODUCT_ATTENTION, SCALED_DOT_PRODUCT_ATTENTION_BACKWARD -> true;
            case MIN_GRAD, MAX_GRAD, REDUCE_MIN_GRAD, REDUCE_MAX_GRAD -> !input.contiguous();
            case MATMUL, LINEAR, CONV2D, CONV2D_GEMM, CONV2D_BACKWARD_INPUT, CONV2D_BACKWARD_WEIGHT,
                    CONV2D_BACKWARD_INPUT_GEMM, CONV2D_BACKWARD_WEIGHT_GEMM,
                    MAX_POOL2D, MAX_POOL2D_BACKWARD_INPUT, AVG_POOL2D, AVG_POOL2D_BACKWARD_INPUT -> true;
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
