package tuning.report;

public record ReportPolicy(
        int hotStepLimit,
        boolean includeTrace,
        boolean includeFailedCandidates
) {
    public ReportPolicy {
        if (hotStepLimit < 0) {
            throw new IllegalArgumentException("hotStepLimit must be >= 0");
        }
    }

    public static ReportPolicy defaults() {
        return new ReportPolicy(20, true, true);
    }
}
