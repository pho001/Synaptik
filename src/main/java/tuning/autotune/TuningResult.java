package tuning.autotune;

import config.profile.ExecutionProfile;
import tuning.benchmark.report.BenchmarkCandidateReport;
import tuning.autotune.report.TuningSummary;

import java.util.List;

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
