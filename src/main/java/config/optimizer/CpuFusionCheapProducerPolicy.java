package config.optimizer;

/**
 * Policy for cheap producers feeding CPU fused loops.
 */
public enum CpuFusionCheapProducerPolicy {
    EXTERNAL_INPUT,
    INLINE_IF_SINGLE_USE,
    INLINE_CHEAP_SHARED
}
