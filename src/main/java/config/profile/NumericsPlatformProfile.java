package config.profile;

import backend.ApproxMode;

import java.util.Objects;

public record NumericsPlatformProfile(
        ApproxMode approxMode,
        boolean forceExactTranscendentals
) {
    public NumericsPlatformProfile {
        approxMode = Objects.requireNonNull(approxMode, "approxMode cannot be null");
    }
}
