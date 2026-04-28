package backend;

/**
 * Policy for approximate transcendental operations.
 *
 * <p>The mode is interpreted by {@link config.runtime.ApproximationConfig}. It does not select kernels
 * directly; execution context creation resolves it into booleans such as "use fast exp".</p>
 */
public enum ApproxMode {
    /**
     * Never use approximate transcendental kernels.
     */
    OFF,
    /**
     * Use approximate kernels only when backward/training execution is enabled.
     */
    TRAINING_ONLY,
    /**
     * Use approximate kernels whenever an approximation implementation is available.
     */
    ALWAYS
}
