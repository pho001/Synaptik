package tuning.search;

import java.util.List;

public record SearchTreeSnapshot(
        List<SearchTreeNode> nodes,
        List<String> frontierFingerprints
) {
    public SearchTreeSnapshot {
        nodes = nodes == null ? List.of() : List.copyOf(nodes);
        frontierFingerprints = frontierFingerprints == null ? List.of() : List.copyOf(frontierFingerprints);
    }
}
