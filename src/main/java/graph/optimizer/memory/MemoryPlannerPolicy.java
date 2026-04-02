package graph.optimizer.memory;

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
}
