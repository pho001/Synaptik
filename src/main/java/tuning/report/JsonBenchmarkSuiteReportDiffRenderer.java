package tuning.report;

import java.util.Locale;

public final class JsonBenchmarkSuiteReportDiffRenderer {
    private JsonBenchmarkSuiteReportDiffRenderer() {
    }

    public static String render(BenchmarkSuiteReportDiff diff) {
        if (diff == null) {
            throw new IllegalArgumentException("diff cannot be null");
        }
        StringBuilder sb = new StringBuilder();
        sb.append("{\n");
        sb.append("  \"previousCreatedAt\": \"").append(diff.previousCreatedAt()).append("\",\n");
        sb.append("  \"currentCreatedAt\": \"").append(diff.currentCreatedAt()).append("\",\n");
        sb.append("  \"previousOverallBest\": \"").append(escape(diff.previousOverallBestCandidate())).append("\",\n");
        sb.append("  \"currentOverallBest\": \"").append(escape(diff.currentOverallBestCandidate())).append("\",\n");
        sb.append("  \"previousTotalSuccesses\": ").append(diff.previousTotalSuccesses()).append(",\n");
        sb.append("  \"currentTotalSuccesses\": ").append(diff.currentTotalSuccesses()).append(",\n");
        sb.append("  \"workloads\": [\n");
        for (int i = 0; i < diff.workloadDiffs().size(); i++) {
            if (i > 0) {
                sb.append(",\n");
            }
            BenchmarkReportDiff workload = diff.workloadDiffs().get(i);
            sb.append("    {");
            sb.append("\"name\": \"").append(escape(workload.workloadName())).append("\", ");
            sb.append("\"previousBest\": \"").append(escape(workload.previousBestCandidate())).append("\", ");
            sb.append("\"currentBest\": \"").append(escape(workload.currentBestCandidate())).append("\", ");
            sb.append("\"currentBestMedianMs\": ").append(format(workload.currentBestMedianMs())).append(", ");
            sb.append("\"bestSpeedupVsPrevious\": ").append(format(workload.bestSpeedupVsPrevious()));
            sb.append("}");
        }
        sb.append("\n  ]\n");
        sb.append("}\n");
        return sb.toString();
    }

    private static String format(double value) {
        return Double.isFinite(value) ? String.format(Locale.US, "%.6f", value) : "null";
    }

    private static String escape(String value) {
        return value == null ? "" : value.replace("\\", "\\\\").replace("\"", "\\\"");
    }
}
