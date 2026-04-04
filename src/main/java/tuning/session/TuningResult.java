package tuning.session;

import config.profile.ExecutionProfile;
import tuning.report.BenchmarkCandidateReport;

import java.util.List;

public record TuningResult(
        ExecutionProfile bestProfile,
        List<BenchmarkCandidateReport> finalists,
        String summary,
        boolean persisted
) {
    public TuningResult {
        finalists = finalists == null ? List.of() : List.copyOf(finalists);
        summary = summary == null ? "" : summary;
    }
}
