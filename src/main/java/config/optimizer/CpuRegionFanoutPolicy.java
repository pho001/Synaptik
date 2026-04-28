package config.optimizer;

/**
 * Policy for CPU region behavior at graph fanout points.
 */
public enum CpuRegionFanoutPolicy {
    /**
     * Materialize values at fanout boundaries.
     */
    MATERIALIZE_AT_FANOUT,

    /**
     * Include cheap elementwise fanout branches when safe.
     */
    INCLUDE_CHEAP_ELEMENTWISE_BRANCHES,

    /**
     * Include fanout branches but split concrete execution units later.
     */
    INCLUDE_AND_SPLIT_EXECUTION_UNITS
}
