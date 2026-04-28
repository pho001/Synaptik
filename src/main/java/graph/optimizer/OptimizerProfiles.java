package graph.optimizer;

/**
 * Convenience accessors for standard optimizer configurations.
 */
public final class OptimizerProfiles {
    private OptimizerProfiles() {}

    /**
     * Returns a configuration with optimizer stages disabled.
     *
     * @return no-optimization configuration
     */
    public static config.optimizer.OptimizerConfig noOptimization() {
        return config.optimizer.OptimizerConfig.noOptimization();
    }

    /**
     * Returns the default training optimizer configuration.
     *
     * @return training defaults
     */
    public static config.optimizer.OptimizerConfig trainingDefaults() {
        return config.optimizer.OptimizerConfig.trainingDefaults();
    }

    /**
     * Returns the default inference optimizer configuration.
     *
     * @return inference defaults
     */
    public static config.optimizer.OptimizerConfig inferenceDefaults() {
        return config.optimizer.OptimizerConfig.inferenceDefaults();
    }
}
