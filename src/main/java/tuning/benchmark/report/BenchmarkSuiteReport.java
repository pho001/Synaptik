package tuning.benchmark.report;

import tuning.benchmark.BenchmarkEntryRole;

import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

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

    public Optional<BenchmarkCandidateReport> overallBestCandidate() {
        return workloadReports.stream()
                .flatMap(report -> report.candidates().stream())
                .filter(BenchmarkCandidateReport::success)
                .filter(report -> report.measurement() != null)
                .filter(report -> !report.baseline())
                .min(Comparator.comparingDouble(report -> report.measurement().steadyStateStats().medianMs()));
    }

    public List<BenchmarkSuiteCandidateSummary> candidateSummaries() {
        Map<String, CandidateAccumulator> grouped = new LinkedHashMap<>();
        for (BenchmarkReport workloadReport : workloadReports) {
            for (BenchmarkCandidateReport candidateReport : workloadReport.candidates()) {
                CandidateAccumulator accumulator = grouped.computeIfAbsent(
                        candidateReport.entry().name(),
                        ignored -> new CandidateAccumulator(candidateReport.entry().name(), candidateReport.entry().role())
                );
                accumulator.workloadCount++;
                if (candidateReport.success() && candidateReport.measurement() != null) {
                    accumulator.successCount++;
                    accumulator.totalMedianMs += candidateReport.measurement().steadyStateStats().medianMs();
                    accumulator.medianSamples++;
                }
                double speedup = workloadReport.speedupVsBaseline(candidateReport);
                if (Double.isFinite(speedup)) {
                    accumulator.totalSpeedupVsBaseline += speedup;
                    accumulator.speedupVsBaselineSamples++;
                }
            }
        }
        return grouped.values().stream()
                .map(CandidateAccumulator::toSummary)
                .sorted(Comparator
                        .comparing(BenchmarkSuiteCandidateSummary::role)
                        .thenComparing(BenchmarkSuiteCandidateSummary::candidateName))
                .toList();
    }

    public List<GpuCoverageSummary> coverageSummaries() {
        List<GpuCoverageSummary> summaries = new ArrayList<>();
        for (BenchmarkReport workloadReport : workloadReports) {
            for (BenchmarkCandidateReport candidateReport : workloadReport.candidates()) {
                if (candidateReport.measurement() == null) {
                    continue;
                }
                GpuCoverageSummary summary = GpuCoverageSummary.fromTrace(candidateReport.measurement().trace());
                if (summary.present()) {
                    summaries.add(summary);
                }
            }
        }
        return summaries;
    }

    public Map<String, GpuCoverageSummary.BackendCoverage> bestCoverageByBackend() {
        Map<String, GpuCoverageSummary.BackendCoverage> best = new LinkedHashMap<>();
        for (GpuCoverageSummary summary : coverageSummaries()) {
            for (var entry : summary.backends().entrySet()) {
                best.merge(entry.getKey(), entry.getValue(), BenchmarkSuiteReport::betterCoverage);
            }
        }
        return best;
    }

    public List<BenchmarkSuiteHotspot> hotspots(int limit) {
        if (limit <= 0) {
            return List.of();
        }
        List<BenchmarkSuiteHotspot> hotspots = new ArrayList<>();
        for (BenchmarkReport workloadReport : workloadReports) {
            for (BenchmarkCandidateReport candidateReport : workloadReport.candidates()) {
                if (candidateReport.measurement() == null) {
                    continue;
                }
                for (trace.execution.ExecutionStepTrace step : candidateReport.measurement().trace().run().steps()) {
                    hotspots.add(new BenchmarkSuiteHotspot(
                            workloadReport.workloadName(),
                            candidateReport.entry().name(),
                            step.opType(),
                            step.label(),
                            step.durationNs()
                    ));
                }
            }
        }
        return hotspots.stream()
                .sorted(Comparator.comparingLong(BenchmarkSuiteHotspot::durationNs).reversed())
                .limit(limit)
                .toList();
    }

    public List<BenchmarkCandidateReport> candidateReports(String candidateName) {
        if (candidateName == null || candidateName.isBlank()) {
            return List.of();
        }
        return workloadReports.stream()
                .flatMap(report -> report.candidates().stream())
                .filter(report -> report.entry() != null)
                .filter(report -> candidateName.equals(report.entry().name()))
                .toList();
    }

    private static GpuCoverageSummary.BackendCoverage betterCoverage(
            GpuCoverageSummary.BackendCoverage left,
            GpuCoverageSummary.BackendCoverage right
    ) {
        int selected = Integer.compare(left.maxSelectedRegionLength(), right.maxSelectedRegionLength());
        if (selected != 0) {
            return selected > 0 ? left : right;
        }
        int materializations = Integer.compare(left.cpuMaterializationCount(), right.cpuMaterializationCount());
        if (materializations != 0) {
            return materializations < 0 ? left : right;
        }
        int fallbacks = Integer.compare(left.fallbackCount(), right.fallbackCount());
        if (fallbacks != 0) {
            return fallbacks < 0 ? left : right;
        }
        int handoffs = Integer.compare(left.deviceHandoffCount(), right.deviceHandoffCount());
        return handoffs <= 0 ? left : right;
    }

    private static final class CandidateAccumulator {
        private final String candidateName;
        private final BenchmarkEntryRole role;
        private long workloadCount;
        private long successCount;
        private double totalMedianMs;
        private int medianSamples;
        private double totalSpeedupVsBaseline;
        private int speedupVsBaselineSamples;

        private CandidateAccumulator(String candidateName, BenchmarkEntryRole role) {
            this.candidateName = candidateName;
            this.role = role;
        }

        private BenchmarkSuiteCandidateSummary toSummary() {
            return new BenchmarkSuiteCandidateSummary(
                    candidateName,
                    role,
                    workloadCount,
                    successCount,
                    medianSamples == 0 ? Double.NaN : totalMedianMs / medianSamples,
                    speedupVsBaselineSamples == 0 ? Double.NaN : totalSpeedupVsBaseline / speedupVsBaselineSamples
            );
        }
    }
}
