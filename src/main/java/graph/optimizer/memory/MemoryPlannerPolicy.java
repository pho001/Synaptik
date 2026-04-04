package graph.optimizer.memory;

import config.optimizer.MemoryConfig;

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

    public static MemoryPlannerPolicy defaults() {
        return new MemoryPlannerPolicy(
                true,
                false,
                false,
                1
        );
    }

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
