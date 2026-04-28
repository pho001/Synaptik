package config.optimizer;

/**
 * Graph-level policy controlling CPU fused-loop selection inside CPU execution regions.
 */
public enum CpuFusionMode {
    OFF,
    LOCAL_CONSERVATIVE,
    LOCAL_BALANCED,
    LOCAL_AGGRESSIVE
}
