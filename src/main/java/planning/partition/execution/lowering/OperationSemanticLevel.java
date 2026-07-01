package planning.partition.execution.lowering;

/**
 * Semantic level of an operation as seen by partition-aware lowering.
 */
public enum OperationSemanticLevel {
    PRIMITIVE,
    CANONICAL_HIGH_LEVEL,
    BACKEND_FRIENDLY_HIGH_LEVEL,
    COMPOSITE,
    TRAINING_BACKWARD,
    LAYOUT,
    FUSED,
    UNKNOWN
}
