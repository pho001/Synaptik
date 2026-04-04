package tuning.report;

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
                .min(Comparator.comparingDouble(report -> report.measurement().steadyStateStats().medianMs()));
    }

    public List<BenchmarkSuiteCandidateSummary> candidateSummaries() {
        Map<String, CandidateAccumulator> grouped = new LinkedHashMap<>();
        for (BenchmarkReport workloadReport : workloadReports) {
            for (BenchmarkCandidateReport candidateReport : workloadReport.candidates()) {
                CandidateAccumulator accumulator = grouped.computeIfAbsent(
                        candidateReport.candidate().name(),
                        ignored -> new CandidateAccumulator(candidateReport.candidate().name(), candidateReport.baselineKind())
                );
                accumulator.workloadCount++;
                if (candidateReport.success() && candidateReport.measurement() != null) {
                    accumulator.successCount++;
                    accumulator.totalMedianMs += candidateReport.measurement().steadyStateStats().medianMs();
                    accumulator.medianSamples++;
                }
                double speedupNoOpt = workloadReport.speedupVsNoOpt(candidateReport);
                if (Double.isFinite(speedupNoOpt)) {
                    accumulator.totalSpeedupVsNoOpt += speedupNoOpt;
                    accumulator.speedupVsNoOptSamples++;
                }
                double speedupNoOptCr = workloadReport.speedupVsNoOptConservativeRuntime(candidateReport);
                if (Double.isFinite(speedupNoOptCr)) {
                    accumulator.totalSpeedupVsNoOptConservativeRuntime += speedupNoOptCr;
                    accumulator.speedupVsNoOptConservativeRuntimeSamples++;
                }
            }
        }
        return grouped.values().stream()
                .map(CandidateAccumulator::toSummary)
                .sorted(Comparator
                        .comparing(BenchmarkSuiteCandidateSummary::baselineKind)
                        .thenComparing(BenchmarkSuiteCandidateSummary::candidateName))
                .toList();
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
                for (graph.execution.trace.ExecutionStepTrace step : candidateReport.measurement().trace().run().steps()) {
                    hotspots.add(new BenchmarkSuiteHotspot(
                            workloadReport.workloadName(),
                            candidateReport.candidate().name(),
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

    private static final class CandidateAccumulator {
        private final String candidateName;
        private final BenchmarkBaselineKind baselineKind;
        private long workloadCount;
        private long successCount;
        private double totalMedianMs;
        private int medianSamples;
        private double totalSpeedupVsNoOpt;
        private int speedupVsNoOptSamples;
        private double totalSpeedupVsNoOptConservativeRuntime;
        private int speedupVsNoOptConservativeRuntimeSamples;

        private CandidateAccumulator(String candidateName, BenchmarkBaselineKind baselineKind) {
            this.candidateName = candidateName;
            this.baselineKind = baselineKind;
        }

        private BenchmarkSuiteCandidateSummary toSummary() {
            return new BenchmarkSuiteCandidateSummary(
                    candidateName,
                    baselineKind,
                    workloadCount,
                    successCount,
                    medianSamples == 0 ? Double.NaN : totalMedianMs / medianSamples,
                    speedupVsNoOptSamples == 0 ? Double.NaN : totalSpeedupVsNoOpt / speedupVsNoOptSamples,
                    speedupVsNoOptConservativeRuntimeSamples == 0
                            ? Double.NaN
                            : totalSpeedupVsNoOptConservativeRuntime / speedupVsNoOptConservativeRuntimeSamples
            );
        }
    }
}
