package backend.metal.lowering;

import backend.ComputeBackend;
import graph.CompiledNode;
import operations.Operation;

public final class MetalPartitionSupport {
    private MetalPartitionSupport() {
    }

    public static boolean isPlannerSupported(CompiledNode node) {
        if (node == null
                || node.backend() != ComputeBackend.GPU_METAL
                || node.operation() == null
                || node.inputIds().isEmpty()) {
            return false;
        }
        if (node.backwardNode()) {
            return switch (node.operation().opType()) {
                case MATMUL, LINEAR, SOFTMAX_GRAD, LOG_SOFTMAX_GRAD, REDUCE_MIN_GRAD, REDUCE_MAX_GRAD, MIN_GRAD, MAX_GRAD, SCALED_DOT_PRODUCT_ATTENTION_BACKWARD -> true;
                default -> false;
            };
        }
        return switch (node.operation().opType()) {
            case MATMUL, LINEAR, ADD, SUB, MUL, DIV, RELU, TANH, FAST_TANH, SIGMOID, ABS, EXP, FAST_EXP, LOG, NEG, SQRT, INV, MUL_SCALAR, WHERE, SOFTMAX, CLAMP_MIN, CLAMP_MAX, RESHAPE, CONTIGUOUS, NOOP, PERMUTE, EXPAND_DIMS, SQUEEZE -> true;
            default -> false;
        };
    }

    public static boolean containsMatMulFamily(CompiledNode node) {
        if (node == null || node.operation() == null) {
            return false;
        }
        Operation.OpType opType = node.operation().opType();
        return opType == Operation.OpType.MATMUL || opType == Operation.OpType.LINEAR;
    }
}
