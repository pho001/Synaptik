package tuning.report;

public record BenchmarkSuiteCandidateSummary(
        String candidateName,
        BenchmarkBaselineKind baselineKind,
        long workloadCount,
        long successCount,
        double averageMedianMs,
        double averageSpeedupVsNoOpt,
        double averageSpeedupVsNoOptConservativeRuntime
) {
    public BenchmarkSuiteCandidateSummary {
        candidateName = candidateName == null ? "" : candidateName;
        baselineKind = baselineKind == null ? BenchmarkBaselineKind.NONE : baselineKind;
    }
}
