package tuning.report;

import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public record BenchmarkSuiteReportDiff(
        OffsetDateTime previousCreatedAt,
        OffsetDateTime currentCreatedAt,
        String previousOverallBestCandidate,
        String currentOverallBestCandidate,
        long previousTotalSuccesses,
        long currentTotalSuccesses,
        List<BenchmarkReportDiff> workloadDiffs
) {
    public BenchmarkSuiteReportDiff {
        workloadDiffs = workloadDiffs == null ? List.of() : List.copyOf(workloadDiffs);
    }

    public static BenchmarkSuiteReportDiff compare(BenchmarkSuiteReport previous, BenchmarkSuiteReport current) {
        if (previous == null || current == null) {
            throw new IllegalArgumentException("previous and current suite reports cannot be null");
        }
        Map<String, BenchmarkReport> previousByWorkload = new LinkedHashMap<>();
        for (BenchmarkReport report : previous.workloadReports()) {
            previousByWorkload.put(report.workloadName(), report);
        }
        Map<String, BenchmarkReport> currentByWorkload = new LinkedHashMap<>();
        for (BenchmarkReport report : current.workloadReports()) {
            currentByWorkload.put(report.workloadName(), report);
        }
        java.util.LinkedHashSet<String> workloadNames = new java.util.LinkedHashSet<>();
        workloadNames.addAll(previousByWorkload.keySet());
        workloadNames.addAll(currentByWorkload.keySet());

        List<BenchmarkReportDiff> diffs = new ArrayList<>(workloadNames.size());
        for (String workloadName : workloadNames) {
            BenchmarkReport before = previousByWorkload.get(workloadName);
            BenchmarkReport after = currentByWorkload.get(workloadName);
            if (before != null && after != null) {
                diffs.add(BenchmarkReportDiff.compare(before, after));
            }
        }

        return new BenchmarkSuiteReportDiff(
                previous.createdAt(),
                current.createdAt(),
                previous.overallBestCandidate().map(c -> c.entry().name()).orElse(""),
                current.overallBestCandidate().map(c -> c.entry().name()).orElse(""),
                previous.totalSuccessCount(),
                current.totalSuccessCount(),
                diffs
        );
    }
}
