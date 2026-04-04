package tuning.search;

public interface SearchTreeInspectable {
    SearchTreeSnapshot snapshot();

    default SearchTreeReport report() {
        SearchTreeSnapshot snapshot = snapshot();
        int maxDepth = snapshot.nodes().stream().mapToInt(SearchTreeNode::depth).max().orElse(0);
        return new SearchTreeReport(
                getClass().getSimpleName(),
                snapshot.nodes().size(),
                snapshot.frontierFingerprints().size(),
                maxDepth,
                snapshot.nodes(),
                snapshot.frontierFingerprints(),
                java.util.Map.of(
                        "nodeCount", snapshot.nodes().size(),
                        "frontierSize", snapshot.frontierFingerprints().size(),
                        "maxDepth", maxDepth
                )
        );
    }
}
