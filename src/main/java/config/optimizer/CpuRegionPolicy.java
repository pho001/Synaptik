package config.optimizer;

/**
 * Graph-level policy controlling CPU execution region construction.
 */
public enum CpuRegionPolicy {
    /**
     * Do not create CPU execution regions.
     */
    OFF,

    /**
     * Create conservative regions around elementwise islands.
     */
    ELEMENTWISE_ISLANDS,

    /**
     * Create natural CPU regions that may include unit-kernel boundaries.
     */
    NATURAL_CPU_REGIONS,

    /**
     * Create wider CPU regions with broader fan-in/fanout tolerance.
     */
    AGGRESSIVE_CPU_REGIONS
}
