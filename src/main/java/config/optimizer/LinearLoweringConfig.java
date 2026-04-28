package config.optimizer;

/**
 * Enables or disables lowering high-level linear operations into primitive matmul/add operations.
 *
 * @param enabled whether linear lowering should run when rewrite lowering is active
 */
public record LinearLoweringConfig(
        boolean enabled
) {
    /**
     * @return default enabled linear lowering config
     */
    public static LinearLoweringConfig defaults() {
        return new LinearLoweringConfig(true);
    }

    /**
     * @return config that disables linear lowering
     */
    public static LinearLoweringConfig disabled() {
        return new LinearLoweringConfig(false);
    }
}
