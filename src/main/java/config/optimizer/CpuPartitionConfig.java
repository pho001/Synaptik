package config.optimizer;

/**
 * Graph optimizer configuration for CPU natural execution partition planning.
 *
 * @param policy high-level CPU partition policy
 * @param maxPartitionNodes maximum nodes in one CPU execution partition
 * @param fanoutPolicy fanout handling policy
 * @param boundaryPolicy boundary inclusion policy
 */
public record CpuPartitionConfig(
        CpuPartitionPolicy policy,
        int maxPartitionNodes,
        CpuPartitionFanoutPolicy fanoutPolicy,
        CpuPartitionBoundaryPolicy boundaryPolicy
) {
    public CpuPartitionConfig {
        policy = policy == null ? CpuPartitionPolicy.NATURAL_CPU_PARTITIONS : policy;
        maxPartitionNodes = Math.max(1, maxPartitionNodes);
        fanoutPolicy = fanoutPolicy == null ? CpuPartitionFanoutPolicy.MATERIALIZE_AT_FANOUT : fanoutPolicy;
        boundaryPolicy = boundaryPolicy == null ? CpuPartitionBoundaryPolicy.INCLUDE_UNIT_BOUNDARIES : boundaryPolicy;
    }

    /**
     * Returns production defaults for CPU natural execution partitions.
     *
     * @return default CPU partition config
     */
    public static CpuPartitionConfig defaults() {
        return new CpuPartitionConfig(
                CpuPartitionPolicy.NATURAL_CPU_PARTITIONS,
                64,
                CpuPartitionFanoutPolicy.MATERIALIZE_AT_FANOUT,
                CpuPartitionBoundaryPolicy.INCLUDE_UNIT_BOUNDARIES
        );
    }

    /**
     * Returns a disabled CPU partition config.
     *
     * @return disabled CPU partition config
     */
    public static CpuPartitionConfig off() {
        return new CpuPartitionConfig(
                CpuPartitionPolicy.OFF,
                1,
                CpuPartitionFanoutPolicy.MATERIALIZE_AT_FANOUT,
                CpuPartitionBoundaryPolicy.ELEMENTWISE_ONLY
        );
    }

    /**
     * Returns an elementwise-island CPU partition config.
     *
     * @return elementwise-island CPU partition config
     */
    public static CpuPartitionConfig elementwiseIslands() {
        return new CpuPartitionConfig(
                CpuPartitionPolicy.ELEMENTWISE_ISLANDS,
                32,
                CpuPartitionFanoutPolicy.MATERIALIZE_AT_FANOUT,
                CpuPartitionBoundaryPolicy.ELEMENTWISE_ONLY
        );
    }

    /**
     * Returns a wider exploratory CPU partition config.
     *
     * @return aggressive CPU partition config
     */
    public static CpuPartitionConfig aggressive() {
        return new CpuPartitionConfig(
                CpuPartitionPolicy.AGGRESSIVE_CPU_PARTITIONS,
                128,
                CpuPartitionFanoutPolicy.INCLUDE_AND_SPLIT_EXECUTION_UNITS,
                CpuPartitionBoundaryPolicy.INCLUDE_UNIT_BOUNDARIES
        );
    }
}
