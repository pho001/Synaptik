package tuning.benchmark.report;

import java.time.OffsetDateTime;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;

public record BenchmarkReport(
        String workloadName,
        OffsetDateTime createdAt,
        List<BenchmarkCandidateReport> candidates,
        String bestCandidateName
) {
    public BenchmarkReport {
        workloadName = workloadName == null ? "workload" : workloadName;
        createdAt = createdAt == null ? OffsetDateTime.now() : createdAt;
        candidates = candidates == null ? List.of() : List.copyOf(candidates);
        bestCandidateName = bestCandidateName == null ? "" : bestCandidateName;
    }

    public static BenchmarkReport of(String workloadName, List<BenchmarkCandidateReport> candidates) {
        String best = candidates == null ? "" : candidates.stream()
                .filter(BenchmarkCandidateReport::success)
                .filter(r -> r.measurement() != null)
                .filter(r -> !r.baseline())
                .min(Comparator.comparingDouble(r -> r.measurement().steadyStateStats().medianMs()))
                .map(r -> r.entry().name())
                .orElse("");
        return new BenchmarkReport(workloadName, OffsetDateTime.now(), candidates, best);
    }

    public long successCount() {
        return candidates.stream().filter(BenchmarkCandidateReport::success).count();
    }

    public long failureCount() {
        return candidates.size() - successCount();
    }

    public Optional<BenchmarkCandidateReport> bestCandidate() {
        return candidates.stream()
                .filter(BenchmarkCandidateReport::success)
                .filter(r -> r.measurement() != null)
                .filter(r -> !r.baseline())
                .min(Comparator.comparingDouble(r -> r.measurement().steadyStateStats().medianMs()));
    }

    public Optional<BenchmarkCandidateReport> baseline() {
        return candidates.stream()
                .filter(BenchmarkCandidateReport::baseline)
                .findFirst();
    }

    public double speedupVsBaseline(BenchmarkCandidateReport candidate) {
        if (candidate == null || candidate.measurement() == null) {
            return Double.NaN;
        }
        return baseline()
                .filter(base -> base.measurement() != null)
                .map(base -> base.measurement().steadyStateStats().medianMs() / candidate.measurement().steadyStateStats().medianMs())
                .orElse(Double.NaN);
    }
}
