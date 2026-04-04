package tuning.report;

import java.time.OffsetDateTime;
import java.util.List;

public record BenchmarkSuiteReport(
        OffsetDateTime createdAt,
        List<BenchmarkReport> workloadReports
) {
    public BenchmarkSuiteReport {
        createdAt = createdAt == null ? OffsetDateTime.now() : createdAt;
        workloadReports = workloadReports == null ? List.of() : List.copyOf(workloadReports);
    }

    public long totalCandidateCount() {
        return workloadReports.stream().mapToLong(report -> report.candidates().size()).sum();
    }

    public long totalSuccessCount() {
        return workloadReports.stream().mapToLong(BenchmarkReport::successCount).sum();
    }

    public long totalFailureCount() {
        return workloadReports.stream().mapToLong(BenchmarkReport::failureCount).sum();
    }
}
