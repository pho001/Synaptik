package tuning.benchmark.report;

import java.util.Locale;

public final class TextBenchmarkSuiteReportRenderer {
    private static final String ANSI_GREEN = "\u001B[32m";
    private static final String ANSI_RESET = "\u001B[0m";

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
        report.overallBestCandidate().ifPresent(best -> sb.append("overallBestCandidate=")
                .append(colorize(best.entry().name()))
                .append('\n'));
        report.overallBestCandidate().ifPresent(best -> sb.append("overallBestMedianMs=")
                .append(String.format(Locale.US, "%.6f", best.measurement().steadyStateStats().medianMs()))
                .append("\n\n"));

        sb.append("Workloads\n");
        sb.append(String.format(
                Locale.US,
                "%-28s %-16s %-12s %-12s%n",
                "name", "bestCandidate", "successes", "failures"
        ));
        for (BenchmarkReport workloadReport : report.workloadReports()) {
            String row = String.format(
                    Locale.US,
                    "%-28s %-16s %-12d %-12d%n",
                    workloadReport.workloadName(),
                    workloadReport.bestCandidateName().isBlank() ? "n/a" : workloadReport.bestCandidateName(),
                    workloadReport.successCount(),
                    workloadReport.failureCount()
            );
            sb.append(colorizeWorkloadRowIfNeeded(row, workloadReport));
        }
        sb.append("\n");

        sb.append("Candidate Summaries\n");
        sb.append(String.format(
                Locale.US,
                "%-34s %-12s %-10s %-10s %-14s %-14s%n",
                "name", "role", "workloads", "successes", "avgMedianMs", "avgVsBaseline"
        ));
        for (BenchmarkSuiteCandidateSummary summary : report.candidateSummaries()) {
            sb.append(String.format(
                    Locale.US,
                    "%-34s %-12s %-10d %-10d %-14s %-14s%n",
                    summary.candidateName(),
                    summary.role().name(),
                    summary.workloadCount(),
                    summary.successCount(),
                    formatDouble(summary.averageMedianMs()),
                    formatRatio(summary.averageSpeedupVsBaseline())
            ));
        }
        sb.append("\n");

        sb.append("coverageSummary:\n");
        for (var entry : report.bestCoverageByBackend().entrySet()) {
            var coverage = entry.getValue();
            sb.append("- backend=").append(entry.getKey())
                    .append(" gpuCoverageRatio=").append(formatDouble(coverage.gpuCoverageRatio()))
                    .append(" maxSelectedRegionLength=").append(coverage.maxSelectedRegionLength())
                    .append(" cpuMaterializationCount=").append(coverage.cpuMaterializationCount())
                    .append(" fallbackCount=").append(coverage.fallbackCount())
                    .append(" deviceHandoffCount=").append(coverage.deviceHandoffCount())
                    .append('\n');
        }
        sb.append("\n");

        sb.append("Suite Hotspots\n");
        for (BenchmarkSuiteHotspot hotspot : report.hotspots(10)) {
            sb.append("- ")
                    .append(hotspot.workloadName())
                    .append(" / ")
                    .append(hotspot.candidateName())
                    .append(" / ")
                    .append(hotspot.opType())
                    .append(" [")
                    .append(hotspot.label())
                    .append("] ")
                    .append(String.format(Locale.US, "%.6fms", hotspot.durationNs() / 1_000_000.0d))
                    .append('\n');
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

    private static String formatDouble(double value) {
        return Double.isFinite(value) ? String.format(Locale.US, "%.6f", value) : "n/a";
    }

    private static String formatRatio(double ratio) {
        return Double.isFinite(ratio) ? String.format(Locale.US, "%.3fx", ratio) : "n/a";
    }

    private static String colorize(String value) {
        if (value == null || value.isBlank()) {
            return value;
        }
        return ANSI_GREEN + value + ANSI_RESET;
    }

    private static String colorizeWorkloadRowIfNeeded(String row, BenchmarkReport workloadReport) {
        if (row == null || row.isEmpty() || workloadReport == null || workloadReport.bestCandidateName().isBlank()) {
            return row;
        }
        return ANSI_GREEN + row + ANSI_RESET;
    }
}
