package config.profile;

import config.backend.SumAccuracyMode;

import java.util.Objects;

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
