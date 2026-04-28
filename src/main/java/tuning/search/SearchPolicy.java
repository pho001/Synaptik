package tuning.search;

/**
 * Bounds candidate search during autotune.
 *
 * <p>{@link #maxCandidates()} limits initial candidate selection,
 * {@link #beamWidth()} limits finalist promotion and beam-like strategies, and
 * {@link #maxRounds()} bounds refinement. Strategies may ignore
 * {@link #allowPruning()} only when their algorithm has no pruning behavior.</p>
 *
 * @param maxCandidates maximum candidates selected by a search batch
 * @param beamWidth number of top candidates retained by beam-style workflows
 * @param maxRounds maximum initial plus refinement rounds
 * @param allowPruning whether strategies may discard candidates by bounds/history
 */
public record SearchPolicy(
        int maxCandidates,
        int beamWidth,
        int maxRounds,
        boolean allowPruning
) {
    public SearchPolicy {
        if (maxCandidates < 1) {
            throw new IllegalArgumentException("maxCandidates must be >= 1");
        }
        if (beamWidth < 1) {
            throw new IllegalArgumentException("beamWidth must be >= 1");
        }
        if (maxRounds < 1) {
            throw new IllegalArgumentException("maxRounds must be >= 1");
        }
    }
}
