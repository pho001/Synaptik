package graph.optimizer;

public final class OptimizerProfiles {
    private OptimizerProfiles() {}

    public static config.optimizer.OptimizerConfig noOptimization() {
        return config.optimizer.OptimizerConfig.noOptimization();
    }

    public static config.optimizer.OptimizerConfig trainingDefaults() {
        return config.optimizer.OptimizerConfig.trainingDefaults();
    }

    public static config.optimizer.OptimizerConfig inferenceDefaults() {
        return config.optimizer.OptimizerConfig.inferenceDefaults();
    }
}
