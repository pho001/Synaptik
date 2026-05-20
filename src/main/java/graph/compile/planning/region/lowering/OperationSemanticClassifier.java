package graph.compile.planning.region.lowering;

import operations.Operation;

/**
 * Backend-neutral operation taxonomy used by partition and region lowering policy.
 */
public final class OperationSemanticClassifier {
    private OperationSemanticClassifier() {
    }

    public static OperationSemanticLevel classify(Operation operation) {
        if (operation == null || operation.opType() == null) {
            return OperationSemanticLevel.UNKNOWN;
        }
        return classify(operation.opType());
    }

    public static OperationSemanticLevel classify(Operation.OpType opType) {
        if (opType == null) {
            return OperationSemanticLevel.UNKNOWN;
        }
        return switch (opType) {
            case ADD, SUB, MUL, DIV, MIN, MAX,
                 GT, GE, LT, LE, EQ, NE,
                 LOGICAL_AND, LOGICAL_OR, LOGICAL_NOT, WHERE,
                 NEG, INV, LOG, EXP, FAST_EXP, ERF, TANH, FAST_TANH,
                 POW, POW_TENSOR, SQRT, ABS, FLOOR, CEIL, SIGN, MUL_SCALAR, RELU, CLAMP_MIN, CLAMP_MAX,
                 SIGMOID, MATMUL, SUM, MEAN, REDUCE_MIN, REDUCE_MAX, REDUCE_PROD, CUMSUM, ARGMAX, REDUCE_ALL, REDUCE_ANY ->
                    OperationSemanticLevel.PRIMITIVE;
            case RESHAPE, EXPAND, SELECT, SLICE, CONCAT, PAD, TILE, PERMUTE, EXPAND_DIMS, SQUEEZE, CONTIGUOUS, NOOP ->
                    OperationSemanticLevel.LAYOUT;
            case LINEAR, LOG_SOFTMAX, SOFTMAX, LAYER_NORM, RMS_NORM ->
                    OperationSemanticLevel.CANONICAL_HIGH_LEVEL;
            case CONV2D, CONV2D_GEMM, MAX_POOL2D, AVG_POOL2D, SCALED_DOT_PRODUCT_ATTENTION,
                 SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS ->
                    OperationSemanticLevel.BACKEND_FRIENDLY_HIGH_LEVEL;
            case NLL_LOSS, CROSS_ENTROPY_LOSS, CROSS_ENTROPY_LOSS_INDICES ->
                    OperationSemanticLevel.COMPOSITE;
            case SOFTMAX_GRAD, LOG_SOFTMAX_GRAD, MIN_GRAD, MAX_GRAD,
                 REDUCE_MIN_GRAD, REDUCE_MAX_GRAD,
                 GATHER_GRAD, GATHER_AXIS_GRAD, GATHER_ND_GRAD, TAKE_ALONG_AXIS_GRAD, SLICE_GRAD,
                 CROSS_ENTROPY_LOSS_INDICES_GRAD,
                 SCALED_DOT_PRODUCT_ATTENTION_BACKWARD,
                 CONV2D_BACKWARD_INPUT, CONV2D_BACKWARD_WEIGHT,
                 CONV2D_BACKWARD_INPUT_GEMM, CONV2D_BACKWARD_WEIGHT_GEMM,
                 MAX_POOL2D_BACKWARD_INPUT, AVG_POOL2D_BACKWARD_INPUT ->
                    OperationSemanticLevel.TRAINING_BACKWARD;
            case GATHER, GATHER_AXIS, GATHER_ND, TAKE_ALONG_AXIS, SCATTER_ADD, SCATTER_AXIS_ADD,
                 SCATTER_ELEMENTS, SCATTER_ND, SLICE_SCATTER_ADD ->
                    OperationSemanticLevel.BACKEND_FRIENDLY_HIGH_LEVEL;
            case FUSED -> OperationSemanticLevel.FUSED;
            case CAST, CONST_SCALAR, UNKNOWN -> OperationSemanticLevel.UNKNOWN;
        };
    }
}
