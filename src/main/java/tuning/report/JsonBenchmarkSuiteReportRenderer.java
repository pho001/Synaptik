package tuning.report;

public final class JsonBenchmarkSuiteReportRenderer {
    private JsonBenchmarkSuiteReportRenderer() {
    }

    public static String render(BenchmarkSuiteReport report) {
        if (report == null) {
            throw new IllegalArgumentException("report cannot be null");
        }
        StringBuilder sb = new StringBuilder();
        sb.append("{\n");
        sb.append("  \"createdAt\": \"").append(report.createdAt()).append("\",\n");
        sb.append("  \"totalCandidates\": ").append(report.totalCandidateCount()).append(",\n");
        sb.append("  \"totalSuccesses\": ").append(report.totalSuccessCount()).append(",\n");
        sb.append("  \"totalFailures\": ").append(report.totalFailureCount()).append(",\n");
        sb.append("  \"workloads\": [\n");
        for (int i = 0; i < report.workloadReports().size(); i++) {
            if (i > 0) {
                sb.append(",\n");
            }
            String nested = JsonBenchmarkReportRenderer.render(report.workloadReports().get(i)).indent(4).stripTrailing();
            sb.append(nested);
        }
        sb.append("\n  ]\n");
        sb.append("}\n");
        return sb.toString();
    }
}
