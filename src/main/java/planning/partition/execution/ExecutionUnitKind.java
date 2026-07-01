package planning.partition.execution;

/**
 * Kind of execution unit produced by partition planning.
 */
public enum ExecutionUnitKind {
    FUSED_ELEMENTWISE,
    SINGLE_OP,
    UNIT_KERNEL,
    SPECIALIZED_PRIMITIVE,
    MATERIALIZATION_BOUNDARY
}
