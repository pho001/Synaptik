package config.compile;

import config.optimizer.CpuRegionConfig;

/**
 * Source-of-truth compile policy split across semantic, graph, backend, region, and memory planning layers.
 */
public record CompileConfig(
        SemanticCanonicalizationConfig semanticCanonicalization,
        GraphOptimizationConfig graphOptimization,
        BackendPlanningConfig backendPlanning,
        RegionOptimizationConfig regionOptimization,
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
        regionOptimization = regionOptimization == null
                ? RegionOptimizationConfig.trainingDefaults()
                : regionOptimization;
        memoryPlanning = memoryPlanning == null ? MemoryPlanningConfig.defaults() : memoryPlanning;
    }

    public static CompileConfig training() {
        return new CompileConfig(
                SemanticCanonicalizationConfig.defaults(),
                GraphOptimizationConfig.trainingDefaults(),
                BackendPlanningConfig.explicitOnly(),
                RegionOptimizationConfig.trainingDefaults(),
                MemoryPlanningConfig.trainingDefaults()
        );
    }

    public static CompileConfig inference() {
        return new CompileConfig(
                SemanticCanonicalizationConfig.defaults(),
                GraphOptimizationConfig.inferenceDefaults(),
                BackendPlanningConfig.explicitOnly(),
                RegionOptimizationConfig.inferenceDefaults(),
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
                .withBackendPlanning(BackendPlanningConfig.cpuOnly().withCpuRegions(CpuRegionConfig.off()))
                .withRegionOptimization(RegionOptimizationConfig.disabled())
                .withMemoryPlanning(MemoryPlanningConfig.disabledUnlessRequired());
    }

    public CompileConfig withSemanticCanonicalization(SemanticCanonicalizationConfig newConfig) {
        return new CompileConfig(newConfig, graphOptimization, backendPlanning, regionOptimization, memoryPlanning);
    }

    public CompileConfig withGraphOptimization(GraphOptimizationConfig newConfig) {
        return new CompileConfig(semanticCanonicalization, newConfig, backendPlanning, regionOptimization, memoryPlanning);
    }

    public CompileConfig withBackendPlanning(BackendPlanningConfig newConfig) {
        return new CompileConfig(semanticCanonicalization, graphOptimization, newConfig, regionOptimization, memoryPlanning);
    }

    public CompileConfig withRegionOptimization(RegionOptimizationConfig newConfig) {
        return new CompileConfig(semanticCanonicalization, graphOptimization, backendPlanning, newConfig, memoryPlanning);
    }

    public CompileConfig withMemoryPlanning(MemoryPlanningConfig newConfig) {
        return new CompileConfig(semanticCanonicalization, graphOptimization, backendPlanning, regionOptimization, newConfig);
    }

    public static CompileConfig defaultsForMode(backend.runtime.ExecutionMode mode) {
        return mode == backend.runtime.ExecutionMode.FORWARD_BACKWARD ? training() : inference();
    }

    public static CompileConfig defaultsForCompileMode(tensor.CompileMode mode, boolean hasTrainableLeafInputs) {
        return switch (mode == null ? tensor.CompileMode.INFERENCE_ONLY : mode) {
            case INFERENCE_ONLY -> inference();
            case TRAINING -> training();
            case AUTO -> hasTrainableLeafInputs ? training() : inference();
        };
    }
}
