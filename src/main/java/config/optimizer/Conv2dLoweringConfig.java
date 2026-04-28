package config.optimizer;

import java.util.Objects;

/**
 * Configuration for conv2d lowering during the rewrite stage.
 *
 * @param mode lowering mode; {@code null} becomes {@link Conv2dLoweringMode#HEURISTIC}
 */
public record Conv2dLoweringConfig(
        Conv2dLoweringMode mode
) {
    public Conv2dLoweringConfig {
        mode = Objects.requireNonNullElse(mode, Conv2dLoweringMode.HEURISTIC);
    }

    /**
     * @return heuristic conv2d lowering config
     */
    public static Conv2dLoweringConfig defaults() {
        return new Conv2dLoweringConfig(Conv2dLoweringMode.HEURISTIC);
    }

    /**
     * @return config that disables conv2d lowering
     */
    public static Conv2dLoweringConfig off() {
        return new Conv2dLoweringConfig(Conv2dLoweringMode.OFF);
    }

    /**
     * @return config that always lowers eligible conv2d operations
     */
    public static Conv2dLoweringConfig always() {
        return new Conv2dLoweringConfig(Conv2dLoweringMode.ALWAYS);
    }
}
