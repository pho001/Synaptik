package tuning.benchmark.report;

public record BenchmarkSuiteHotspot(
        String workloadName,
        String candidateName,
        String opType,
        String label,
        long durationNs
) {
    public BenchmarkSuiteHotspot {
        workloadName = workloadName == null ? "" : workloadName;
        candidateName = candidateName == null ? "" : candidateName;
        opType = opType == null ? "" : opType;
        label = label == null ? "" : label;
    }
}
