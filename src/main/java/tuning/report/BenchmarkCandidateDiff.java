package tuning.report;

import tuning.session.BenchmarkEntryRole;

public record BenchmarkCandidateDiff(
        String candidateName,
        BenchmarkEntryRole role,
        boolean previousSuccess,
        boolean currentSuccess,
        double previousMedianMs,
        double currentMedianMs,
        double deltaMedianMs,
        double speedupVsPrevious
) {
    public BenchmarkCandidateDiff {
        candidateName = candidateName == null ? "" : candidateName;
        role = role == null ? BenchmarkEntryRole.CANDIDATE : role;
    }
}
