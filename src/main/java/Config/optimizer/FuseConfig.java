package Config.optimizer;

public final class FuseConfig {
    private final int maxClusterNodes;
    private final double scoreThreshold;
    private final double internalEdgeBonus;
    private final double externalInputPenalty;
    private final double sharedExpensivePenalty;
    private final double nonCheapBonus;
    private final boolean preserveSharedExpensiveNodes;

    public FuseConfig(
            int maxClusterNodes,
            double scoreThreshold,
            double internalEdgeBonus,
            double externalInputPenalty,
            double sharedExpensivePenalty,
            double nonCheapBonus,
            boolean preserveSharedExpensiveNodes
    ) {
        this.maxClusterNodes = maxClusterNodes;
        this.scoreThreshold = scoreThreshold;
        this.internalEdgeBonus = internalEdgeBonus;
        this.externalInputPenalty = externalInputPenalty;
        this.sharedExpensivePenalty = sharedExpensivePenalty;
        this.nonCheapBonus = nonCheapBonus;
        this.preserveSharedExpensiveNodes = preserveSharedExpensiveNodes;
    }

    public int maxClusterNodes() {
        return maxClusterNodes;
    }

    public double scoreThreshold() {
        return scoreThreshold;
    }

    public double internalEdgeBonus() {
        return internalEdgeBonus;
    }

    public double externalInputPenalty() {
        return externalInputPenalty;
    }

    public double sharedExpensivePenalty() {
        return sharedExpensivePenalty;
    }

    public double nonCheapBonus() {
        return nonCheapBonus;
    }

    public boolean preserveSharedExpensiveNodes() {
        return preserveSharedExpensiveNodes;
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

    public static FuseConfig trainingDefaults() {
        return new FuseConfig(
                64,
                0.55,
                0.25,
                0.20,
                1.00,
                0.30,
                true
        );
    }

    public static FuseConfig inferencePerfDefaults() {
        return new FuseConfig(
                96,
                0.6,
                0.30,
                0.10,
                0.50,
                0.35,
                false
        );
    }
}
