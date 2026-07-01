package config.optimizer;

/**
 * Scoring and safety configuration for partition fusion.
 *
 * <p>Fusion groups compatible operations into larger execution units. The score parameters trade off
 * loop-combining benefit against extra inputs, shared expensive nodes, and graph complexity. All numeric
 * score parameters must be finite and non-negative.</p>
 *
 * @param maxClusterNodes maximum number of nodes in one fused cluster
 * @param scoreThreshold minimum score required to accept a fusion candidate
 * @param internalEdgeBonus bonus for edges internal to the candidate
 * @param externalInputPenalty penalty for extra external inputs
 * @param sharedExpensivePenalty penalty for consuming nodes that are expensive and shared elsewhere
 * @param nonCheapBonus bonus for fusing non-cheap operations where loop reduction matters more
 * @param preserveSharedExpensiveNodes whether shared expensive nodes should remain outside fused partitions
 */
public record FuseConfig(
        int maxClusterNodes,
        double scoreThreshold,
        double internalEdgeBonus,
        double externalInputPenalty,
        double sharedExpensivePenalty,
        double nonCheapBonus,
        boolean preserveSharedExpensiveNodes
) {
    public FuseConfig {
        if (maxClusterNodes <= 0) {
            throw new IllegalArgumentException("maxClusterNodes must be > 0");
        }
        if (Double.isNaN(scoreThreshold) || Double.isInfinite(scoreThreshold)) {
            throw new IllegalArgumentException("scoreThreshold must be finite");
        }
        if (Double.isNaN(internalEdgeBonus) || Double.isInfinite(internalEdgeBonus)) {
            throw new IllegalArgumentException("internalEdgeBonus must be finite");
        }
        if (Double.isNaN(externalInputPenalty) || Double.isInfinite(externalInputPenalty)) {
            throw new IllegalArgumentException("externalInputPenalty must be finite");
        }
        if (Double.isNaN(sharedExpensivePenalty) || Double.isInfinite(sharedExpensivePenalty)) {
            throw new IllegalArgumentException("sharedExpensivePenalty must be finite");
        }
        if (Double.isNaN(nonCheapBonus) || Double.isInfinite(nonCheapBonus)) {
            throw new IllegalArgumentException("nonCheapBonus must be finite");
        }
        if (scoreThreshold < 0.0d) {
            throw new IllegalArgumentException("scoreThreshold must be >= 0");
        }
        if (internalEdgeBonus < 0.0d) {
            throw new IllegalArgumentException("internalEdgeBonus must be >= 0");
        }
        if (externalInputPenalty < 0.0d) {
            throw new IllegalArgumentException("externalInputPenalty must be >= 0");
        }
        if (sharedExpensivePenalty < 0.0d) {
            throw new IllegalArgumentException("sharedExpensivePenalty must be >= 0");
        }
        if (nonCheapBonus < 0.0d) {
            throw new IllegalArgumentException("nonCheapBonus must be >= 0");
        }
    }

    /**
     * @return conservative fusion defaults for training graphs
     */
    public static FuseConfig trainingDefaults() {
        return new FuseConfig(
                64,
                0.55d,
                0.30d,
                0.20d,
                1.00d,
                0.35d,
                true
        );
    }

    /**
     * @return more permissive fusion defaults for inference graphs
     */
    public static FuseConfig inferenceDefaults() {
        return new FuseConfig(
                96,
                0.00d,
                0.50d,
                0.10d,
                0.50d,
                0.35d,
                false
        );
    }

    /**
     * @return alias for inference defaults used by performance profiles
     */
    public static FuseConfig inferencePerfDefaults() {
        return inferenceDefaults();
    }

    public FuseConfig withMaxClusterNodes(int value) {
        return new FuseConfig(
                value,
                scoreThreshold,
                internalEdgeBonus,
                externalInputPenalty,
                sharedExpensivePenalty,
                nonCheapBonus,
                preserveSharedExpensiveNodes
        );
    }

    public FuseConfig withScoreThreshold(double value) {
        return new FuseConfig(
                maxClusterNodes,
                value,
                internalEdgeBonus,
                externalInputPenalty,
                sharedExpensivePenalty,
                nonCheapBonus,
                preserveSharedExpensiveNodes
        );
    }

    public FuseConfig withInternalEdgeBonus(double value) {
        return new FuseConfig(
                maxClusterNodes,
                scoreThreshold,
                value,
                externalInputPenalty,
                sharedExpensivePenalty,
                nonCheapBonus,
                preserveSharedExpensiveNodes
        );
    }

    public FuseConfig withExternalInputPenalty(double value) {
        return new FuseConfig(
                maxClusterNodes,
                scoreThreshold,
                internalEdgeBonus,
                value,
                sharedExpensivePenalty,
                nonCheapBonus,
                preserveSharedExpensiveNodes
        );
    }

    public FuseConfig withSharedExpensivePenalty(double value) {
        return new FuseConfig(
                maxClusterNodes,
                scoreThreshold,
                internalEdgeBonus,
                externalInputPenalty,
                value,
                nonCheapBonus,
                preserveSharedExpensiveNodes
        );
    }

    public FuseConfig withNonCheapBonus(double value) {
        return new FuseConfig(
                maxClusterNodes,
                scoreThreshold,
                internalEdgeBonus,
                externalInputPenalty,
                sharedExpensivePenalty,
                value,
                preserveSharedExpensiveNodes
        );
    }

    public FuseConfig withPreserveSharedExpensiveNodes(boolean value) {
        return new FuseConfig(
                maxClusterNodes,
                scoreThreshold,
                internalEdgeBonus,
                externalInputPenalty,
                sharedExpensivePenalty,
                nonCheapBonus,
                value
        );
    }
}
