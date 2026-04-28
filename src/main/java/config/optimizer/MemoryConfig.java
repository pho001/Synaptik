package config.optimizer;

/**
 * Memory planner reuse policy.
 *
 * <p>The memory planner can reuse intermediate buffers when lifetimes do not overlap. This config
 * controls whether forward/backward pools are separated, whether reuse may cross phases, whether larger
 * buffers may satisfy smaller requests, and the minimum size that is worth considering for reuse.</p>
 *
 * @param separateForwardBackwardPools whether forward and backward buffers are planned in separate pools
 * @param allowCrossPhaseReuse whether a buffer may be reused across forward/backward phase boundaries
 * @param allowLargerBufferReuse whether a larger reusable buffer may satisfy a smaller allocation
 * @param minReusableBufferSize minimum allocation size considered for reuse
 */
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

    /**
     * @return default conservative memory planner config
     */
    public static MemoryConfig defaults() {
        return new MemoryConfig(
                true,
                false,
                false,
                1
        );
    }

    /**
     * @param value replacement separate-pools setting
     * @return updated memory config
     */
    public MemoryConfig withSeparateForwardBackwardPools(boolean value) {
        return new MemoryConfig(value, allowCrossPhaseReuse, allowLargerBufferReuse, minReusableBufferSize);
    }

    /**
     * @param value replacement cross-phase reuse setting
     * @return updated memory config
     */
    public MemoryConfig withAllowCrossPhaseReuse(boolean value) {
        return new MemoryConfig(separateForwardBackwardPools, value, allowLargerBufferReuse, minReusableBufferSize);
    }

    /**
     * @param value replacement larger-buffer reuse setting
     * @return updated memory config
     */
    public MemoryConfig withAllowLargerBufferReuse(boolean value) {
        return new MemoryConfig(separateForwardBackwardPools, allowCrossPhaseReuse, value, minReusableBufferSize);
    }

    /**
     * @param value replacement minimum reusable buffer size
     * @return updated memory config
     */
    public MemoryConfig withMinReusableBufferSize(int value) {
        return new MemoryConfig(separateForwardBackwardPools, allowCrossPhaseReuse, allowLargerBufferReuse, value);
    }
}
