package tuning.search;

import java.util.List;
import java.util.Map;

public record SearchTreeReport(
        String strategyName,
        int nodeCount,
        int frontierSize,
        int maxDepth,
        List<SearchTreeNode> nodes,
        List<String> frontierFingerprints,
        Map<String, Object> summary
) {
    public SearchTreeReport {
        strategyName = strategyName == null ? "search" : strategyName;
        nodes = nodes == null ? List.of() : List.copyOf(nodes);
        frontierFingerprints = frontierFingerprints == null ? List.of() : List.copyOf(frontierFingerprints);
        summary = summary == null ? Map.of() : Map.copyOf(summary);
    }
}
