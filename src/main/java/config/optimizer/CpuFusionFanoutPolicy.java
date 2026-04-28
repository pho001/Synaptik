package config.optimizer;

/**
 * Policy for CPU fused-loop behavior at fanout points.
 */
public enum CpuFusionFanoutPolicy {
    STOP_AT_FANOUT,
    ALLOW_CHEAP_DUPLICATION,
    MATERIALIZE_AND_CONTINUE
}
