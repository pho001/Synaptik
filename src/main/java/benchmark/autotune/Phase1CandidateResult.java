package benchmark.autotune;

public record Phase1CandidateResult(
        Status status,
        CandidatePerf perf,
        CorrectnessVerdict safetyVerdict,
        CorrectnessVerdict fullVerdict,
        String unsafeReason
) {
    public enum Status {
        SKIPPED_UNSAFE_HISTORY,
        MISMATCH_SAFETY,
        SAFE_SWEEP,
        MISMATCH_FULL,
        VALID_PHASE1
    }
}
