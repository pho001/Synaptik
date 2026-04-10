package config.profile;

import config.optimizer.OptimizerConfig;

import java.util.Objects;

public record GraphExecutionPolicy(
        OptimizerConfig optimizer
) {
    public GraphExecutionPolicy {
        optimizer = Objects.requireNonNull(optimizer, "optimizer cannot be null");
    }

    public static GraphExecutionPolicy of(OptimizerConfig optimizer) {
        return new GraphExecutionPolicy(optimizer);
    }

    public static GraphExecutionPolicy fromExecutionProfile(ExecutionProfile profile) {
        Objects.requireNonNull(profile, "profile cannot be null");
        return new GraphExecutionPolicy(profile.optimizer());
    }

    public static GraphExecutionPolicy trainingDefaults() {
        return new GraphExecutionPolicy(OptimizerConfig.trainingDefaults());
    }

    public static GraphExecutionPolicy inferenceDefaults() {
        return new GraphExecutionPolicy(OptimizerConfig.inferenceDefaults());
    }

    public static GraphExecutionPolicy noOptimization() {
        return new GraphExecutionPolicy(OptimizerConfig.noOptimization());
    }
}
