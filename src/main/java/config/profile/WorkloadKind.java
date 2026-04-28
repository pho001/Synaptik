package config.profile;

/**
 * Coarse workload category attached to execution profiles.
 *
 * <p>Runtime and graph policies can use this value to distinguish generic execution from specialized
 * hot-path workloads whose shape patterns justify separate thresholds or candidates.</p>
 */
public enum WorkloadKind {
    /**
     * No specialized workload information is attached.
     */
    NONE,

    /**
     * Transformer-style attention/feed-forward hot path with explicit batch/head/sequence dimensions.
     */
    TRANSFORMER_HOT_PATH
}
