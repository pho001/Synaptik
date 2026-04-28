package tuning.benchmark.report;

import java.time.OffsetDateTime;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;

/**
 * Immutable report for one benchmark workload.
 *
 * <p>The best candidate is the successful, non-baseline entry with the lowest
 * steady-state median latency. A missing baseline makes speedup calculations
 * return {@link Double#NaN}.</p>
 *
 * @param workloadName workload name used in reports
 * @param createdAt report creation time; {@code null} becomes now
 * @param candidates per-entry reports; {@code null} becomes empty
 * @param bestCandidateName cached best candidate name, or blank when none
 */
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

    /**
     * Builds a report and derives the best-candidate name from measured results.
     *
     * @param workloadName workload name
     * @param candidates candidate reports
     * @return report with derived best candidate name
     */
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

    /**
     * @return count of successful candidate reports
     */
    public long successCount() {
        return candidates.stream().filter(BenchmarkCandidateReport::success).count();
    }

    /**
     * @return count of failed candidate reports
     */
    public long failureCount() {
        return candidates.size() - successCount();
    }

    /**
     * @return fastest successful non-baseline candidate, if any
     */
    public Optional<BenchmarkCandidateReport> bestCandidate() {
        return candidates.stream()
                .filter(BenchmarkCandidateReport::success)
                .filter(r -> r.measurement() != null)
                .filter(r -> !r.baseline())
                .min(Comparator.comparingDouble(r -> r.measurement().steadyStateStats().medianMs()));
    }

    /**
     * @return first baseline report, if supplied
     */
    public Optional<BenchmarkCandidateReport> baseline() {
        return candidates.stream()
                .filter(BenchmarkCandidateReport::baseline)
                .findFirst();
    }

    /**
     * Computes relative speedup against the baseline median latency.
     *
     * @param candidate candidate report to compare
     * @return baseline median divided by candidate median, or {@link Double#NaN}
     * if either side is unavailable
     */
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
