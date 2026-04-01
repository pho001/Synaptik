package backend.runtime;

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

    public boolean useFastExp(boolean backwardEnabled) {
        if (forceExactTranscendentals) {
            return false;
        }
        return switch (approxMode) {
            case OFF -> false;
            case ALWAYS -> true;
            case TRAINING_ONLY -> backwardEnabled;
        };
    }

    public boolean useFastTanh(boolean backwardEnabled) {
        if (forceExactTranscendentals) {
            return false;
        }
        return switch (approxMode) {
            case OFF -> false;
            case ALWAYS -> true;
            case TRAINING_ONLY -> backwardEnabled;
        };
    }
}
