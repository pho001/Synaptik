package benchmark.autotune;

import benchmark.OptimizerCandidate;

import java.util.List;

public record FinalistPreparationResult(Status status, List<OptimizerCandidate> finalists) {
    public enum Status {
        OK,
        EMPTY_AFTER_POSTCHECK
    }
}
