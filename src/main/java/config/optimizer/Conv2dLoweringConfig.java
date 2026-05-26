package config.optimizer;

import java.util.Objects;

/**
 * Configuration for conv2d lowering during the rewrite stage.
 *
 * @param mode lowering mode; {@code null} becomes {@link Conv2dLoweringMode#HEURISTIC}
 * @param profile tunable heuristic thresholds; {@code null} uses conservative defaults
 */
public record Conv2dLoweringConfig(
        Conv2dLoweringMode mode,
        Conv2dDagLoweringProfile profile
) {
    public Conv2dLoweringConfig {
        mode = Objects.requireNonNullElse(mode, Conv2dLoweringMode.HEURISTIC);
        profile = profile == null ? Conv2dDagLoweringProfile.defaults() : profile;
    }

    public Conv2dLoweringConfig(Conv2dLoweringMode mode) {
        this(mode, Conv2dDagLoweringProfile.defaults());
    }

    /**
     * @return heuristic conv2d lowering config
     */
    public static Conv2dLoweringConfig defaults() {
        return new Conv2dLoweringConfig(Conv2dLoweringMode.HEURISTIC, Conv2dDagLoweringProfile.defaults());
    }

    /**
     * @return config that disables conv2d lowering
     */
    public static Conv2dLoweringConfig off() {
        return new Conv2dLoweringConfig(Conv2dLoweringMode.OFF, Conv2dDagLoweringProfile.defaults());
    }

    /**
     * @return config that always lowers eligible conv2d operations
     */
    public static Conv2dLoweringConfig always() {
        return new Conv2dLoweringConfig(Conv2dLoweringMode.ALWAYS, Conv2dDagLoweringProfile.defaults());
    }
}
