package config.optimizer;

/**
 * Policy for layout/view operations encountered by CPU fused-loop planning.
 */
public enum CpuFusionLayoutPolicy {
    BOUNDARY,
    ALIAS_VIEW_PASSTHROUGH
}
