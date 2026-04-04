package config.optimizer;

import java.util.Objects;

public record Conv2dLoweringConfig(
        Conv2dLoweringMode mode
) {
    public Conv2dLoweringConfig {
        mode = Objects.requireNonNullElse(mode, Conv2dLoweringMode.HEURISTIC);
    }

    public static Conv2dLoweringConfig defaults() {
        return new Conv2dLoweringConfig(Conv2dLoweringMode.HEURISTIC);
    }

    public static Conv2dLoweringConfig off() {
        return new Conv2dLoweringConfig(Conv2dLoweringMode.OFF);
    }

    public static Conv2dLoweringConfig always() {
        return new Conv2dLoweringConfig(Conv2dLoweringMode.ALWAYS);
    }
}
