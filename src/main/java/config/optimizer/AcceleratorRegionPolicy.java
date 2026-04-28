package config.optimizer;

/**
 * Graph-level policy for accelerator ownership region construction.
 */
public enum AcceleratorRegionPolicy {
    /**
     * Do not create accelerator ownership regions.
     */
    OFF,

    /**
     * Use deterministic greedy closed-region planning for accelerator targets.
     */
    GREEDY_CLOSED_REGIONS,

    /**
     * Use scored closed-region search for accelerator targets.
     */
    SCORED_PROFITABLE_REGIONS
}
