package planning.region.lowering;

/**
 * Semantic level of an operation as seen by region-aware lowering.
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
