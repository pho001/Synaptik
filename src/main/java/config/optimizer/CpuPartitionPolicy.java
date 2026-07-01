package config.optimizer;

/**
 * Graph-level policy controlling CPU execution partition construction.
 */
public enum CpuPartitionPolicy {
    /**
     * Do not create CPU execution partitions.
     */
    OFF,

    /**
     * Create conservative partitions around elementwise islands.
     */
    ELEMENTWISE_ISLANDS,

    /**
     * Create natural CPU partitions that may include unit-kernel boundaries.
     */
    NATURAL_CPU_PARTITIONS,

    /**
     * Create wider CPU partitions with broader fan-in/fanout tolerance.
     */
    AGGRESSIVE_CPU_PARTITIONS
}
