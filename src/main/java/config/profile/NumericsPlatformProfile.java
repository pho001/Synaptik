package config.profile;

import backend.ApproxMode;

import java.util.Objects;

/**
 * Numerical approximation policy stored inside a platform runtime profile.
 *
 * <p>This section is copied to {@link config.runtime.ApproximationConfig}. It is persisted with runtime
 * profiles so calibrated artifacts fully describe how numeric kernels should execute, but it is not a
 * normal timing-selected calibration family because approximation changes can affect numerical
 * semantics.</p>
 *
 * @param approxMode approximation mode for fast transcendental implementations
 * @param forceExactTranscendentals whether transcendental operations must use exact implementations
 */
public record NumericsPlatformProfile(
        ApproxMode approxMode,
        boolean forceExactTranscendentals
) {
    public NumericsPlatformProfile {
        approxMode = Objects.requireNonNull(approxMode, "approxMode cannot be null");
    }
}
