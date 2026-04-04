package tuning.report;

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
                .min(Comparator.comparingDouble(r -> r.measurement().steadyStateStats().medianMs()))
                .map(r -> r.candidate().name())
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
                .min(Comparator.comparingDouble(r -> r.measurement().steadyStateStats().medianMs()));
    }
}
