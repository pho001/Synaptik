package tuning.report;

public final class TextBenchmarkSuiteReportRenderer {
    private TextBenchmarkSuiteReportRenderer() {
    }

    public static String render(BenchmarkSuiteReport report) {
        if (report == null) {
            throw new IllegalArgumentException("report cannot be null");
        }

        StringBuilder sb = new StringBuilder();
        sb.append("Benchmark Suite Report\n");
        sb.append("createdAt=").append(report.createdAt()).append("\n");
        sb.append("workloads=").append(report.workloadReports().size()).append("\n\n");
        sb.append("Summary\n");
        sb.append("totalCandidates=").append(report.totalCandidateCount()).append('\n');
        sb.append("totalSuccesses=").append(report.totalSuccessCount()).append('\n');
        sb.append("totalFailures=").append(report.totalFailureCount()).append("\n\n");

        sb.append("Workloads\n");
        sb.append(String.format(
                java.util.Locale.US,
                "%-28s %-16s %-12s %-12s%n",
                "name", "bestCandidate", "successes", "failures"
        ));
        for (BenchmarkReport workloadReport : report.workloadReports()) {
            sb.append(String.format(
                    java.util.Locale.US,
                    "%-28s %-16s %-12d %-12d%n",
                    workloadReport.workloadName(),
                    workloadReport.bestCandidateName().isBlank() ? "n/a" : workloadReport.bestCandidateName(),
                    workloadReport.successCount(),
                    workloadReport.failureCount()
            ));
        }
        sb.append("\n");

        for (int i = 0; i < report.workloadReports().size(); i++) {
            BenchmarkReport workloadReport = report.workloadReports().get(i);
            if (i > 0) {
                sb.append("\n");
            }
            sb.append("=== ").append(workloadReport.workloadName()).append(" ===\n");
            sb.append(TextBenchmarkReportRenderer.render(workloadReport));
        }

        return sb.toString();
    }
}
