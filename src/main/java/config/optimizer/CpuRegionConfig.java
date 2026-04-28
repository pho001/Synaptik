package config.optimizer;

/**
 * Graph optimizer configuration for CPU natural execution region planning.
 *
 * @param policy high-level CPU region policy
 * @param maxRegionNodes maximum nodes in one CPU execution region
 * @param fanoutPolicy fanout handling policy
 * @param boundaryPolicy boundary inclusion policy
 */
public record CpuRegionConfig(
        CpuRegionPolicy policy,
        int maxRegionNodes,
        CpuRegionFanoutPolicy fanoutPolicy,
        CpuRegionBoundaryPolicy boundaryPolicy
) {
    public CpuRegionConfig {
        policy = policy == null ? CpuRegionPolicy.NATURAL_CPU_REGIONS : policy;
        maxRegionNodes = Math.max(1, maxRegionNodes);
        fanoutPolicy = fanoutPolicy == null ? CpuRegionFanoutPolicy.MATERIALIZE_AT_FANOUT : fanoutPolicy;
        boundaryPolicy = boundaryPolicy == null ? CpuRegionBoundaryPolicy.INCLUDE_UNIT_BOUNDARIES : boundaryPolicy;
    }

    /**
     * Returns production defaults for CPU natural execution regions.
     *
     * @return default CPU region config
     */
    public static CpuRegionConfig defaults() {
        return new CpuRegionConfig(
                CpuRegionPolicy.NATURAL_CPU_REGIONS,
                64,
                CpuRegionFanoutPolicy.MATERIALIZE_AT_FANOUT,
                CpuRegionBoundaryPolicy.INCLUDE_UNIT_BOUNDARIES
        );
    }

    /**
     * Returns a disabled CPU region config.
     *
     * @return disabled CPU region config
     */
    public static CpuRegionConfig off() {
        return new CpuRegionConfig(
                CpuRegionPolicy.OFF,
                1,
                CpuRegionFanoutPolicy.MATERIALIZE_AT_FANOUT,
                CpuRegionBoundaryPolicy.ELEMENTWISE_ONLY
        );
    }

    /**
     * Returns an elementwise-island CPU region config.
     *
     * @return elementwise-island CPU region config
     */
    public static CpuRegionConfig elementwiseIslands() {
        return new CpuRegionConfig(
                CpuRegionPolicy.ELEMENTWISE_ISLANDS,
                32,
                CpuRegionFanoutPolicy.MATERIALIZE_AT_FANOUT,
                CpuRegionBoundaryPolicy.ELEMENTWISE_ONLY
        );
    }

    /**
     * Returns a wider exploratory CPU region config.
     *
     * @return aggressive CPU region config
     */
    public static CpuRegionConfig aggressive() {
        return new CpuRegionConfig(
                CpuRegionPolicy.AGGRESSIVE_CPU_REGIONS,
                128,
                CpuRegionFanoutPolicy.INCLUDE_AND_SPLIT_EXECUTION_UNITS,
                CpuRegionBoundaryPolicy.INCLUDE_UNIT_BOUNDARIES
        );
    }
}
