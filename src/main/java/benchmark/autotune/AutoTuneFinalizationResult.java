package benchmark.autotune;

import benchmark.OptimizerCandidate;

import java.util.List;

public record AutoTuneFinalizationResult(
        Status status,
        List<OptimizerCandidate> finalists,
        List<RefinedCandidate> refinedRows,
        AutoTuneBestResults bestResults
) {
    public enum Status {
        NO_VALID_CANDIDATE,
        EMPTY_AFTER_POSTCHECK,
        OK
    }
}
