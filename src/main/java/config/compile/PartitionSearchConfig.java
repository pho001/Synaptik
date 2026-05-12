package config.compile;

/**
 * Search and scoring limits for backend ownership planning.
 */
public record PartitionSearchConfig(
        int maxSearchNodes,
        int maxVisitedCandidates,
        PartitionScoreWeights scoreWeights
) {
    public PartitionSearchConfig {
        maxSearchNodes = Math.max(1, maxSearchNodes);
        maxVisitedCandidates = Math.max(1, maxVisitedCandidates);
        scoreWeights = scoreWeights == null ? PartitionScoreWeights.defaults() : scoreWeights;
    }

    public static PartitionSearchConfig defaults() {
        return new PartitionSearchConfig(64, 512, PartitionScoreWeights.defaults());
    }
}
