package config.runtime;

import backend.ApproxMode;

import java.util.Objects;

public record ApproximationConfig(
        ApproxMode approxMode,
        boolean forceExactTranscendentals
) {
    public ApproximationConfig {
        approxMode = Objects.requireNonNullElse(approxMode, ApproxMode.OFF);
    }

    public static ApproximationConfig defaults() {
        return new ApproximationConfig(ApproxMode.OFF, false);
    }

    public backend.runtime.ApproximationConfig toBackendRuntimeConfig() {
        return new backend.runtime.ApproximationConfig(approxMode, forceExactTranscendentals);
    }

    public static ApproximationConfig fromBackendRuntimeConfig(backend.runtime.ApproximationConfig config) {
        if (config == null) {
            return defaults();
        }
        return new ApproximationConfig(config.approxMode(), config.forceExactTranscendentals());
    }
}
