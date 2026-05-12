package graph.optimizer.region.lowering;

/**
 * Region-aware lowering action selected for an operation or unit.
 */
public enum RegionLoweringAction {
    KEEP_AS_BACKEND_PRIMITIVE,
    LOWER_TO_BACKEND_DAG,
    FUSE_WITH_NEIGHBORS,
    LOWER_WITH_ALTERNATIVES,
    REJECT_WITH_REASON
}
