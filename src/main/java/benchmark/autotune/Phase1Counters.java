package benchmark.autotune;

public record Phase1Counters(
        int processed,
        int valid,
        int mismatch,
        int mismatchSafety,
        int mismatchFull,
        int skippedUnsafe,
        int safetySweepSafe
) {
    public static Phase1Counters zero() {
        return new Phase1Counters(0, 0, 0, 0, 0, 0, 0);
    }

    public Phase1Counters advance(Phase1CandidateResult.Status status) {
        return switch (status) {
            case SKIPPED_UNSAFE_HISTORY -> new Phase1Counters(processed + 1, valid, mismatch, mismatchSafety, mismatchFull, skippedUnsafe + 1, safetySweepSafe);
            case MISMATCH_SAFETY -> new Phase1Counters(processed + 1, valid, mismatch + 1, mismatchSafety + 1, mismatchFull, skippedUnsafe, safetySweepSafe);
            case SAFE_SWEEP -> new Phase1Counters(processed + 1, valid, mismatch, mismatchSafety, mismatchFull, skippedUnsafe, safetySweepSafe + 1);
            case MISMATCH_FULL -> new Phase1Counters(processed + 1, valid, mismatch + 1, mismatchSafety, mismatchFull + 1, skippedUnsafe, safetySweepSafe);
            case VALID_PHASE1 -> new Phase1Counters(processed + 1, valid + 1, mismatch, mismatchSafety, mismatchFull, skippedUnsafe, safetySweepSafe);
        };
    }
}
