package planning.partition;

/**
 * Semantic kind of a planned partition.
 */
public enum PartitionKind {
    /**
     * CPU scheduling/fusion/memory-planning partition. It may contain multiple concrete execution units.
     */
    CPU_EXECUTION,

    /**
     * Accelerator/backend ownership partition. The target backend owns execution of covered nodes.
     */
    ACCELERATOR_OWNERSHIP
}
