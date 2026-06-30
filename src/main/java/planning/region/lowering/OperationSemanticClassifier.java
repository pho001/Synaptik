package planning.region.lowering;

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
        OperationSemanticLevel identityLevel = classifyByIdentity(operation.opType());
        if (identityLevel != null) {
            return identityLevel;
        }
        return classifyByTraits(operation);
    }

    public static OperationSemanticLevel classify(Operation.OpType opType) {
        if (opType == null) {
            return OperationSemanticLevel.UNKNOWN;
        }
        OperationSemanticLevel identityLevel = classifyByIdentity(opType);
        return identityLevel == null ? OperationSemanticLevel.UNKNOWN : identityLevel;
    }

    private static OperationSemanticLevel classifyByIdentity(Operation.OpType opType) {
        return switch (opType) {
            case LINEAR, LOG_SOFTMAX, SOFTMAX, LAYER_NORM, RMS_NORM ->
                    OperationSemanticLevel.CANONICAL_HIGH_LEVEL;
            case CONV2D, MAX_POOL2D, AVG_POOL2D, SCALED_DOT_PRODUCT_ATTENTION,
                 SCALED_DOT_PRODUCT_ATTENTION_WEIGHTS ->
                    OperationSemanticLevel.BACKEND_FRIENDLY_HIGH_LEVEL;
            case NLL_LOSS, CROSS_ENTROPY_LOSS, CROSS_ENTROPY_LOSS_INDICES ->
                    OperationSemanticLevel.COMPOSITE;
            case SOFTMAX_GRAD, LOG_SOFTMAX_GRAD, MIN_GRAD, MAX_GRAD,
                 REDUCE_MIN_GRAD, REDUCE_MAX_GRAD,
                 GATHER_GRAD, GATHER_AXIS_GRAD, GATHER_ND_GRAD, TAKE_ALONG_AXIS_GRAD, SLICE_BACKWARD,
                 CROSS_ENTROPY_LOSS_INDICES_GRAD,
                 SCALED_DOT_PRODUCT_ATTENTION_BACKWARD ->
                    OperationSemanticLevel.TRAINING_BACKWARD;
            case GATHER, GATHER_AXIS, GATHER_ND, TAKE_ALONG_AXIS, SCATTER_ADD, SCATTER_AXIS_ADD,
                 SCATTER_ELEMENTS, SCATTER_ND ->
                    OperationSemanticLevel.BACKEND_FRIENDLY_HIGH_LEVEL;
            case FUSED -> OperationSemanticLevel.FUSED;
            case NOOP -> OperationSemanticLevel.LAYOUT;
            case CAST, CONST_SCALAR, UNKNOWN -> OperationSemanticLevel.UNKNOWN;
            default -> null;
        };
    }

    private static OperationSemanticLevel classifyByTraits(Operation operation) {
        return switch (operation.semanticFamily()) {
            case ARITHMETIC, TRANSCENDENTAL, COMPARISON, LOGICAL, SELECTION, REDUCTION, LINEAR_ALGEBRA ->
                    OperationSemanticLevel.PRIMITIVE;
            case LAYOUT -> OperationSemanticLevel.LAYOUT;
            case FUSED -> OperationSemanticLevel.FUSED;
            case SPECIAL, UNKNOWN -> OperationSemanticLevel.UNKNOWN;
        };
    }

}
