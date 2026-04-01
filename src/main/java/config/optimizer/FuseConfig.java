package config.optimizer;

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
