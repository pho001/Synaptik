package tuning.autotune;

import tuning.autotune.TuningResult;

import java.util.Objects;

public record GraphAutotuneResult(
        GraphAutotuneMode mode,
        TuningResult tuningResult
) {
    public GraphAutotuneResult {
        mode = mode == null ? GraphAutotuneMode.STANDARD : mode;
        Objects.requireNonNull(tuningResult, "tuningResult cannot be null");
    }
}
