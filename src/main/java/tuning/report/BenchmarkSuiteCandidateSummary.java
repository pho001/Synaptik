package tuning.report;

import tuning.session.BenchmarkEntryRole;

public record BenchmarkSuiteCandidateSummary(
        String candidateName,
        BenchmarkEntryRole role,
        long workloadCount,
        long successCount,
        double averageMedianMs,
        double averageSpeedupVsBaseline
) {
    public BenchmarkSuiteCandidateSummary {
        candidateName = candidateName == null ? "" : candidateName;
        role = role == null ? BenchmarkEntryRole.CANDIDATE : role;
    }
}
