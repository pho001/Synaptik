package config.runtime;

import backend.ApproxMode;

import java.util.Objects;

/**
 * Runtime policy for approximate transcendental kernels.
 *
 * <p>Approximation mode controls whether operations such as exp/tanh may use fast approximations.
 * {@code forceExactTranscendentals} is a hard override: when it is true, exact implementations are used
 * regardless of {@link #approxMode()}.</p>
 *
 * @param approxMode approximation mode; {@code null} becomes {@link ApproxMode#OFF}
 * @param forceExactTranscendentals whether exact transcendental implementations are required
 */
public record ApproximationConfig(
        ApproxMode approxMode,
        boolean forceExactTranscendentals
) {
    public ApproximationConfig {
        approxMode = Objects.requireNonNullElse(approxMode, ApproxMode.OFF);
    }

    /**
     * @return default exact-transcendental policy
     */
    public static ApproximationConfig defaults() {
        return new ApproximationConfig(ApproxMode.OFF, false);
    }

    /**
     * Decides whether exp may use the fast approximation path.
     *
     * @param backwardEnabled whether the graph is executing with backward/training support
     * @return {@code true} if fast exp is allowed for the current mode
     */
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

    /**
     * Decides whether tanh may use the fast approximation path.
     *
     * @param backwardEnabled whether the graph is executing with backward/training support
     * @return {@code true} if fast tanh is allowed for the current mode
     */
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
