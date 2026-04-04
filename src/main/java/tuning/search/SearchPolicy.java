package tuning.search;

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
