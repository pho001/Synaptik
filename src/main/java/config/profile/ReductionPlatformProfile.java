package config.profile;

import config.backend.SumAccuracyMode;

import java.util.Objects;

/**
 * Calibrated runtime thresholds for reductions and attention reductions.
 *
 * <p>Reductions have different cost and memory-access patterns from elementwise kernels, so their
 * vector and parallel thresholds are stored separately. The sum accuracy mode is included because it
 * changes reduction implementation choice, but calibration should treat it as a numerical policy rather
 * than a pure timing knob unless a caller explicitly opts into that experiment.</p>
 *
 * @param reductionVectorMinSize minimum element count before vectorizing normal reductions
 * @param reductionParallelMinSize minimum element count before parallelizing normal reductions
 * @param attentionVectorMinSize minimum element count before vectorizing attention reductions
 * @param attentionParallelMinSize minimum element count before parallelizing attention reductions
 * @param sumAccuracyMode numerical policy for sum accumulation
 */
public record ReductionPlatformProfile(
        int reductionVectorMinSize,
        int reductionParallelMinSize,
        int attentionVectorMinSize,
        int attentionParallelMinSize,
        SumAccuracyMode sumAccuracyMode
) {
    public ReductionPlatformProfile {
        sumAccuracyMode = Objects.requireNonNull(sumAccuracyMode, "sumAccuracyMode cannot be null");
    }
}
