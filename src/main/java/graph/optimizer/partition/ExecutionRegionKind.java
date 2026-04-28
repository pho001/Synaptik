package graph.optimizer.partition;

/**
 * Semantic kind of a planned execution region.
 */
public enum ExecutionRegionKind {
    /**
     * CPU scheduling/fusion/memory-planning region. It may contain multiple concrete execution units.
     */
    CPU_EXECUTION,

    /**
     * Accelerator/backend ownership region. The target backend owns execution of covered nodes.
     */
    ACCELERATOR_OWNERSHIP
}
