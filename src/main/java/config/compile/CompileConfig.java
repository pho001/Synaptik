package config.compile;

import config.optimizer.CpuPartitionConfig;

/**
 * Source-of-truth compile policy split across semantic, graph, backend, partition, and memory planning layers.
 */
public record CompileConfig(
        SemanticCanonicalizationConfig semanticCanonicalization,
        GraphOptimizationConfig graphOptimization,
        BackendPlanningConfig backendPlanning,
        PartitionExecutionConfig partitionExecution,
        MemoryPlanningConfig memoryPlanning
) {
    public CompileConfig {
        semanticCanonicalization = semanticCanonicalization == null
                ? SemanticCanonicalizationConfig.defaults()
                : semanticCanonicalization;
        graphOptimization = graphOptimization == null
                ? GraphOptimizationConfig.trainingDefaults()
                : graphOptimization;
        backendPlanning = backendPlanning == null ? BackendPlanningConfig.cpuOnly() : backendPlanning;
        partitionExecution = partitionExecution == null
                ? PartitionExecutionConfig.trainingDefaults()
                : partitionExecution;
        memoryPlanning = memoryPlanning == null ? MemoryPlanningConfig.defaults() : memoryPlanning;
    }

    public static CompileConfig training() {
        return new CompileConfig(
                SemanticCanonicalizationConfig.defaults(),
                GraphOptimizationConfig.trainingDefaults(),
                BackendPlanningConfig.explicitOnly(),
                PartitionExecutionConfig.trainingDefaults(),
                MemoryPlanningConfig.trainingDefaults()
        );
    }

    public static CompileConfig inference() {
        return new CompileConfig(
                SemanticCanonicalizationConfig.defaults(),
                GraphOptimizationConfig.inferenceDefaults(),
                BackendPlanningConfig.explicitOnly(),
                PartitionExecutionConfig.inferenceDefaults(),
                MemoryPlanningConfig.defaults()
        );
    }

    public static CompileConfig trainingAutoAccelerator() {
        return training().withBackendPlanning(BackendPlanningConfig.autoAccelerator());
    }

    public static CompileConfig inferenceAutoAccelerator() {
        return inference().withBackendPlanning(BackendPlanningConfig.autoAccelerator());
    }

    public static CompileConfig trainingExplicitAccelerator() {
        return training().withBackendPlanning(BackendPlanningConfig.explicitOnly());
    }

    public static CompileConfig inferenceExplicitAccelerator() {
        return inference().withBackendPlanning(BackendPlanningConfig.explicitOnly());
    }

    public static CompileConfig requireExplicitAccelerator() {
        return training().withBackendPlanning(BackendPlanningConfig.requireAllExplicitIntents());
    }

    public static CompileConfig noGraphOptimization() {
        return training().withGraphOptimization(GraphOptimizationConfig.noGraphOptimization());
    }

    public static CompileConfig noGraphOptimizationBaseline() {
        return training()
                .withGraphOptimization(GraphOptimizationConfig.noGraphOptimization())
                .withBackendPlanning(BackendPlanningConfig.explicitOnly());
    }

    public static CompileConfig cpuOnlyBaseline() {
        return training()
                .withGraphOptimization(GraphOptimizationConfig.noGraphOptimization())
                .withBackendPlanning(BackendPlanningConfig.cpuOnly().withCpuPartitions(CpuPartitionConfig.off()))
                .withPartitionExecution(PartitionExecutionConfig.disabled())
                .withMemoryPlanning(MemoryPlanningConfig.disabledUnlessRequired());
    }

    public CompileConfig withSemanticCanonicalization(SemanticCanonicalizationConfig newConfig) {
        return new CompileConfig(newConfig, graphOptimization, backendPlanning, partitionExecution, memoryPlanning);
    }

    public CompileConfig withGraphOptimization(GraphOptimizationConfig newConfig) {
        return new CompileConfig(semanticCanonicalization, newConfig, backendPlanning, partitionExecution, memoryPlanning);
    }

    public CompileConfig withBackendPlanning(BackendPlanningConfig newConfig) {
        return new CompileConfig(semanticCanonicalization, graphOptimization, newConfig, partitionExecution, memoryPlanning);
    }

    public CompileConfig withPartitionExecution(PartitionExecutionConfig newConfig) {
        return new CompileConfig(semanticCanonicalization, graphOptimization, backendPlanning, newConfig, memoryPlanning);
    }

    public CompileConfig withMemoryPlanning(MemoryPlanningConfig newConfig) {
        return new CompileConfig(semanticCanonicalization, graphOptimization, backendPlanning, partitionExecution, newConfig);
    }

    public static CompileConfig defaultsForMode(runtime.contract.ExecutionMode mode) {
        return mode == runtime.contract.ExecutionMode.FORWARD_BACKWARD ? training() : inference();
    }

    public static CompileConfig defaultsForCompileMode(tensor.CompileMode mode, boolean hasTrainableLeafInputs) {
        return switch (mode == null ? tensor.CompileMode.INFERENCE_ONLY : mode) {
            case INFERENCE_ONLY -> inference();
            case TRAINING -> training();
            case AUTO -> hasTrainableLeafInputs ? training() : inference();
        };
    }
}
