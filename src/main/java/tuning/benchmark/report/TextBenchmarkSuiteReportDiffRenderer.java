package tuning.benchmark.report;

import java.util.Locale;

public final class TextBenchmarkSuiteReportDiffRenderer {
    private TextBenchmarkSuiteReportDiffRenderer() {
    }

    public static String render(BenchmarkSuiteReportDiff diff) {
        if (diff == null) {
            throw new IllegalArgumentException("diff cannot be null");
        }
        StringBuilder sb = new StringBuilder();
        sb.append("Benchmark Suite Diff\n");
        sb.append("previousCreatedAt=").append(diff.previousCreatedAt()).append('\n');
        sb.append("currentCreatedAt=").append(diff.currentCreatedAt()).append('\n');
        sb.append("previousOverallBest=").append(blankToNa(diff.previousOverallBestCandidate())).append('\n');
        sb.append("currentOverallBest=").append(blankToNa(diff.currentOverallBestCandidate())).append('\n');
        sb.append("previousTotalSuccesses=").append(diff.previousTotalSuccesses()).append('\n');
        sb.append("currentTotalSuccesses=").append(diff.currentTotalSuccesses()).append("\n\n");

        sb.append("Workloads\n");
        sb.append(String.format(
                Locale.US,
                "%-28s %-18s %-18s %-14s %-10s%n",
                "name", "previousBest", "currentBest", "currentBestMs", "speedup"
        ));
        for (BenchmarkReportDiff workload : diff.workloadDiffs()) {
            sb.append(String.format(
                    Locale.US,
                    "%-28s %-18s %-18s %-14s %-10s%n",
                    workload.workloadName(),
                    blankToNa(workload.previousBestCandidate()),
                    blankToNa(workload.currentBestCandidate()),
                    formatDouble(workload.currentBestMedianMs()),
                    formatRatio(workload.bestSpeedupVsPrevious())
            ));
        }
        return sb.toString();
    }

    private static String formatDouble(double value) {
        return Double.isFinite(value) ? String.format(Locale.US, "%.6f", value) : "n/a";
    }

    private static String formatRatio(double value) {
        return Double.isFinite(value) ? String.format(Locale.US, "%.3fx", value) : "n/a";
    }

    private static String blankToNa(String value) {
        return value == null || value.isBlank() ? "n/a" : value;
    }
}
