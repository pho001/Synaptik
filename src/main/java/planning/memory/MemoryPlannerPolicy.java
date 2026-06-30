package planning.memory;

import config.optimizer.MemoryConfig;

/**
 * Policy knobs for tensor storage reuse.
 *
 * @param separateForwardBackwardPools whether forward and backward reusable intervals use separate slot pools
 * @param allowCrossPhaseReuse whether backward work may reuse forward slots
 * @param allowLargerBufferReuse whether a slot may serve a smaller reusable interval
 * @param minReusableBufferSize minimum element count eligible for reuse
 */
public record MemoryPlannerPolicy(
        boolean separateForwardBackwardPools,
        boolean allowCrossPhaseReuse,
        boolean allowLargerBufferReuse,
        int minReusableBufferSize
) {
    public MemoryPlannerPolicy {
        if (minReusableBufferSize < 1) {
            throw new IllegalArgumentException("minReusableBufferSize must be >= 1");
        }
        if (separateForwardBackwardPools && allowCrossPhaseReuse) {
            throw new IllegalArgumentException("allowCrossPhaseReuse cannot be enabled when pools are separated");
        }
    }

    /**
     * Returns the default conservative memory reuse policy.
     *
     * @return default policy
     */
    public static MemoryPlannerPolicy defaults() {
        return new MemoryPlannerPolicy(
                true,
                false,
                false,
                1
        );
    }

    /**
     * Converts optimizer memory configuration to planner policy.
     *
     * @param config memory configuration, or {@code null} for defaults
     * @return planner policy
     */
    public static MemoryPlannerPolicy fromConfig(MemoryConfig config) {
        if (config == null) {
            return defaults();
        }
        return new MemoryPlannerPolicy(
                config.separateForwardBackwardPools(),
                config.allowCrossPhaseReuse(),
                config.allowLargerBufferReuse(),
                config.minReusableBufferSize()
        );
    }
}
