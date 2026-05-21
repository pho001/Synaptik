package graph.compile.planning.region;

/**
 * Kind of execution unit produced by region optimization.
 */
public enum ExecutionUnitKind {
    FUSED_ELEMENTWISE,
    SINGLE_OP,
    UNIT_KERNEL,
    BACKEND_GRAPH,
    SPECIALIZED_PRIMITIVE,
    MATERIALIZATION_BOUNDARY
}
