package tuning.search;

import java.util.Comparator;

public final class TextSearchTreeReportRenderer {
    private TextSearchTreeReportRenderer() {
    }

    public static String render(SearchTreeReport report) {
        if (report == null) {
            throw new IllegalArgumentException("report cannot be null");
        }
        StringBuilder sb = new StringBuilder();
        sb.append("Search Tree Report\n");
        sb.append("strategy=").append(report.strategyName()).append('\n');
        sb.append("nodeCount=").append(report.nodeCount()).append('\n');
        sb.append("frontierSize=").append(report.frontierSize()).append('\n');
        sb.append("maxDepth=").append(report.maxDepth()).append("\n\n");

        sb.append("Frontier\n");
        for (String fp : report.frontierFingerprints()) {
            sb.append("- ").append(fp).append('\n');
        }
        sb.append('\n');

        sb.append("Nodes\n");
        sb.append(String.format("%-10s %-18s %-8s %-8s %-64s%n", "depth", "candidate", "round", "parent?", "fingerprint"));
        report.nodes().stream()
                .sorted(Comparator.comparingInt(SearchTreeNode::depth).thenComparing(SearchTreeNode::candidateName))
                .forEach(node -> sb.append(String.format(
                        "%-10d %-18s %-8d %-8s %-64s%n",
                        node.depth(),
                        node.candidateName(),
                        node.roundDiscovered(),
                        node.parentFingerprint() == null ? "root" : "child",
                        node.fingerprint()
                )));
        return sb.toString();
    }
}
