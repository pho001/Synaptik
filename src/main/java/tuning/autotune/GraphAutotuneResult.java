package tuning.autotune;

import tuning.autotune.TuningResult;

import java.util.Objects;

/**
 * Result wrapper for {@link GraphAutotuneSession}.
 *
 * @param mode graph autotune mode used to generate graph candidates
 * @param tuningResult generic autotune result with finalists and selected profile
 */
public record GraphAutotuneResult(
        GraphAutotuneMode mode,
        TuningResult tuningResult
) {
    public GraphAutotuneResult {
        mode = mode == null ? GraphAutotuneMode.STANDARD : mode;
        Objects.requireNonNull(tuningResult, "tuningResult cannot be null");
    }
}
