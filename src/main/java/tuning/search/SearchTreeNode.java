package tuning.search;

public record SearchTreeNode(
        String fingerprint,
        String candidateName,
        String parentFingerprint,
        int depth,
        int roundDiscovered
) {
}
