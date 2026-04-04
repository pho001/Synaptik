package config.optimizer;

public record LinearLoweringConfig(
        boolean enabled
) {
    public static LinearLoweringConfig defaults() {
        return new LinearLoweringConfig(true);
    }

    public static LinearLoweringConfig disabled() {
        return new LinearLoweringConfig(false);
    }
}
