package config.optimizer;

/**
 * Graph optimizer configuration for CPU fused-loop planning inside CPU execution regions.
 *
 * @param mode high-level fused-loop mode
 * @param maxChainNodes maximum nodes in one fused loop
 * @param fanoutPolicy fused-loop fanout policy
 * @param layoutPolicy layout/view handling policy
 * @param cheapProducerPolicy cheap producer handling policy
 */
public record CpuFusionConfig(
        CpuFusionMode mode,
        int maxChainNodes,
        CpuFusionFanoutPolicy fanoutPolicy,
        CpuFusionLayoutPolicy layoutPolicy,
        CpuFusionCheapProducerPolicy cheapProducerPolicy
) {
    public CpuFusionConfig {
        mode = mode == null ? CpuFusionMode.LOCAL_BALANCED : mode;
        maxChainNodes = Math.max(1, maxChainNodes);
        fanoutPolicy = fanoutPolicy == null ? CpuFusionFanoutPolicy.STOP_AT_FANOUT : fanoutPolicy;
        layoutPolicy = layoutPolicy == null ? CpuFusionLayoutPolicy.BOUNDARY : layoutPolicy;
        cheapProducerPolicy = cheapProducerPolicy == null
                ? CpuFusionCheapProducerPolicy.INLINE_IF_SINGLE_USE
                : cheapProducerPolicy;
    }

    /**
     * Returns default CPU fused-loop policy.
     *
     * @return default CPU fusion config
     */
    public static CpuFusionConfig defaults() {
        return new CpuFusionConfig(
                CpuFusionMode.LOCAL_BALANCED,
                16,
                CpuFusionFanoutPolicy.STOP_AT_FANOUT,
                CpuFusionLayoutPolicy.BOUNDARY,
                CpuFusionCheapProducerPolicy.INLINE_IF_SINGLE_USE
        );
    }

    /**
     * Returns disabled CPU fusion config.
     *
     * @return disabled CPU fusion config
     */
    public static CpuFusionConfig off() {
        return new CpuFusionConfig(
                CpuFusionMode.OFF,
                1,
                CpuFusionFanoutPolicy.STOP_AT_FANOUT,
                CpuFusionLayoutPolicy.BOUNDARY,
                CpuFusionCheapProducerPolicy.EXTERNAL_INPUT
        );
    }

    /**
     * Returns conservative CPU fusion config.
     *
     * @return conservative CPU fusion config
     */
    public static CpuFusionConfig conservative() {
        return new CpuFusionConfig(
                CpuFusionMode.LOCAL_CONSERVATIVE,
                8,
                CpuFusionFanoutPolicy.STOP_AT_FANOUT,
                CpuFusionLayoutPolicy.BOUNDARY,
                CpuFusionCheapProducerPolicy.INLINE_IF_SINGLE_USE
        );
    }

    /**
     * Returns aggressive CPU fusion config.
     *
     * @return aggressive CPU fusion config
     */
    public static CpuFusionConfig aggressive() {
        return new CpuFusionConfig(
                CpuFusionMode.LOCAL_AGGRESSIVE,
                32,
                CpuFusionFanoutPolicy.MATERIALIZE_AND_CONTINUE,
                CpuFusionLayoutPolicy.ALIAS_VIEW_PASSTHROUGH,
                CpuFusionCheapProducerPolicy.INLINE_CHEAP_SHARED
        );
    }
}
