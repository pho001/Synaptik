package config.compile;

/**
 * Score weights used by backend ownership candidate search.
 */
public record PartitionScoreWeights(
        double nodeWeight,
        double internalEdgeWeight,
        double mergeNodeBonus,
        double tailDepthWeight,
        double externalInputPenalty,
        double workWeight
) {
    public PartitionScoreWeights {
        validateFinite("nodeWeight", nodeWeight);
        validateFinite("internalEdgeWeight", internalEdgeWeight);
        validateFinite("mergeNodeBonus", mergeNodeBonus);
        validateFinite("tailDepthWeight", tailDepthWeight);
        validateFinite("externalInputPenalty", externalInputPenalty);
        validateFinite("workWeight", workWeight);
    }

    public static PartitionScoreWeights defaults() {
        return new PartitionScoreWeights(
                1000.0,
                120.0,
                450.0,
                80.0,
                60.0,
                1.0
        );
    }

    private static void validateFinite(String name, double value) {
        if (Double.isNaN(value) || Double.isInfinite(value)) {
            throw new IllegalArgumentException(name + " must be finite");
        }
    }
}
