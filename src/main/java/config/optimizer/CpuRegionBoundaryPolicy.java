package config.optimizer;

/**
 * Policy for which non-elementwise operations may remain inside CPU execution regions.
 */
public enum CpuRegionBoundaryPolicy {
    /**
     * Keep CPU regions limited to elementwise islands.
     */
    ELEMENTWISE_ONLY,

    /**
     * Include operations such as matmul, reductions, and layout operations as unit boundaries.
     */
    INCLUDE_UNIT_BOUNDARIES,

    /**
     * Include safe layout/view passthrough operations when correctness constraints are satisfied.
     */
    INCLUDE_SAFE_LAYOUT_PASSTHROUGH
}
