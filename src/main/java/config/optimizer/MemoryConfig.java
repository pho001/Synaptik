package config.optimizer;

public record MemoryConfig(
        boolean separateForwardBackwardPools,
        boolean allowCrossPhaseReuse,
        boolean allowLargerBufferReuse,
        int minReusableBufferSize
) {
    public MemoryConfig {
        if (minReusableBufferSize < 1) {
            throw new IllegalArgumentException("minReusableBufferSize must be >= 1");
        }
        if (separateForwardBackwardPools && allowCrossPhaseReuse) {
            throw new IllegalArgumentException("allowCrossPhaseReuse cannot be enabled when pools are separated");
        }
    }

    public static MemoryConfig defaults() {
        return new MemoryConfig(
                true,
                false,
                false,
                1
        );
    }

    public MemoryConfig withSeparateForwardBackwardPools(boolean value) {
        return new MemoryConfig(value, allowCrossPhaseReuse, allowLargerBufferReuse, minReusableBufferSize);
    }

    public MemoryConfig withAllowCrossPhaseReuse(boolean value) {
        return new MemoryConfig(separateForwardBackwardPools, value, allowLargerBufferReuse, minReusableBufferSize);
    }

    public MemoryConfig withAllowLargerBufferReuse(boolean value) {
        return new MemoryConfig(separateForwardBackwardPools, allowCrossPhaseReuse, value, minReusableBufferSize);
    }

    public MemoryConfig withMinReusableBufferSize(int value) {
        return new MemoryConfig(separateForwardBackwardPools, allowCrossPhaseReuse, allowLargerBufferReuse, value);
    }
}
