package tuning.autotune;

import config.profile.ExecutionProfile;
import tuning.benchmark.report.BenchmarkCandidateReport;
import tuning.autotune.report.TuningSummary;

import java.util.List;

/**
 * Result of an {@link AutotuneSession} run.
 *
 * <p>The result is immutable and safe to share between threads. A {@code null}
 * {@link #bestProfile()} means no candidate both validated and measured
 * successfully. {@link #persisted()} reports whether the session wrote the best
 * profile, not whether history entries were written.</p>
 *
 * @param bestProfile fastest selected execution profile, or {@code null} when no
 * candidate was usable
 * @param finalists successful candidates ordered by measured median latency
 * @param summary concise human-readable run summary
 * @param details structured counts and best-score details
 * @param persisted whether best-profile persistence succeeded during the run
 */
public record TuningResult(
        ExecutionProfile bestProfile,
        List<BenchmarkCandidateReport> finalists,
        String summary,
        TuningSummary details,
        boolean persisted
) {
    public TuningResult {
        finalists = finalists == null ? List.of() : List.copyOf(finalists);
        summary = summary == null ? "" : summary;
        details = details == null ? new TuningSummary("search", 0, 0, 0, 0, 0, Double.POSITIVE_INFINITY) : details;
    }

    public TuningResult(
            ExecutionProfile bestProfile,
            List<BenchmarkCandidateReport> finalists,
            String summary,
            boolean persisted
    ) {
        this(bestProfile, finalists, summary, null, persisted);
    }
}
