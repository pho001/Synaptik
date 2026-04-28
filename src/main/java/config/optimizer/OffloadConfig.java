package config.optimizer;

/**
 * Graph optimizer configuration for accelerator/offload region planning.
 *
 * @param policy whether accelerator ownership regions may be planned
 * @param acceleratorRegionPolicy strategy preset used when accelerator planning is enabled
 */
public record OffloadConfig(
        OffloadPolicy policy,
        AcceleratorRegionPolicy acceleratorRegionPolicy
) {
    public OffloadConfig {
        policy = policy == null ? OffloadPolicy.CPU_ONLY : policy;
        acceleratorRegionPolicy = acceleratorRegionPolicy == null
                ? AcceleratorRegionPolicy.OFF
                : acceleratorRegionPolicy;
        if (policy == OffloadPolicy.CPU_ONLY) {
            acceleratorRegionPolicy = AcceleratorRegionPolicy.OFF;
        }
    }

    /**
     * Returns conservative CPU-only defaults.
     *
     * @return default offload config
     */
    public static OffloadConfig defaults() {
        return new OffloadConfig(OffloadPolicy.CPU_ONLY, AcceleratorRegionPolicy.OFF);
    }

    /**
     * Returns a policy that allows greedy accelerator ownership regions.
     *
     * @return accelerator-enabled offload config
     */
    public static OffloadConfig acceleratorGreedy() {
        return new OffloadConfig(OffloadPolicy.ACCELERATOR_IF_PROFITABLE, AcceleratorRegionPolicy.GREEDY_CLOSED_REGIONS);
    }

    /**
     * Returns a policy that allows scored accelerator ownership regions.
     *
     * @return accelerator-enabled scored offload config
     */
    public static OffloadConfig acceleratorScored() {
        return new OffloadConfig(OffloadPolicy.ACCELERATOR_IF_PROFITABLE, AcceleratorRegionPolicy.SCORED_PROFITABLE_REGIONS);
    }
}
