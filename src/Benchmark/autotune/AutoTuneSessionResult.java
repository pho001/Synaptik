package Benchmark.autotune;

public record AutoTuneSessionResult(
        Status status,
        Phase1Counters counters,
        Phase1LoopResult phase1,
        AutoTuneFinalizationResult finalization,
        AutoTuneProfilePersistenceResult persistenceResult
) {
    public enum Status {
        SAFE_SWEEP_DONE,
        NO_VALID_CANDIDATE,
        EMPTY_AFTER_POSTCHECK,
        DONE
    }

    public AutoTuneBestResults bestResults() {
        return finalization == null ? null : finalization.bestResults();
    }
}
