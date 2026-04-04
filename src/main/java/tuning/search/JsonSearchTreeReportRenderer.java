package tuning.search;

public final class JsonSearchTreeReportRenderer {
    private JsonSearchTreeReportRenderer() {
    }

    public static String render(SearchTreeReport report) {
        if (report == null) {
            throw new IllegalArgumentException("report cannot be null");
        }
        StringBuilder sb = new StringBuilder();
        sb.append("{\n");
        sb.append("  \"strategyName\": \"").append(report.strategyName()).append("\",\n");
        sb.append("  \"nodeCount\": ").append(report.nodeCount()).append(",\n");
        sb.append("  \"frontierSize\": ").append(report.frontierSize()).append(",\n");
        sb.append("  \"maxDepth\": ").append(report.maxDepth()).append(",\n");
        sb.append("  \"frontierFingerprints\": [");
        for (int i = 0; i < report.frontierFingerprints().size(); i++) {
            if (i > 0) {
                sb.append(", ");
            }
            sb.append("\"").append(report.frontierFingerprints().get(i)).append("\"");
        }
        sb.append("],\n");
        sb.append("  \"nodes\": [\n");
        for (int i = 0; i < report.nodes().size(); i++) {
            SearchTreeNode node = report.nodes().get(i);
            if (i > 0) {
                sb.append(",\n");
            }
            sb.append("    {");
            sb.append("\"fingerprint\": \"").append(node.fingerprint()).append("\", ");
            sb.append("\"candidateName\": \"").append(node.candidateName()).append("\", ");
            sb.append("\"parentFingerprint\": ").append(node.parentFingerprint() == null ? "null" : "\"" + node.parentFingerprint() + "\"").append(", ");
            sb.append("\"depth\": ").append(node.depth()).append(", ");
            sb.append("\"roundDiscovered\": ").append(node.roundDiscovered());
            sb.append("}");
        }
        sb.append("\n  ]\n");
        sb.append("}\n");
        return sb.toString();
    }
}
