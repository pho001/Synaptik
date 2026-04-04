package config.optimizer;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Objects;

public record OptimizerConfig(
        List<OptimizerStage> stageOrder,
        CseConfig cse,
        FuseConfig fuse,
        MemoryConfig memory
) {
    public OptimizerConfig {
        Objects.requireNonNull(stageOrder, "stageOrder cannot be null");
        Objects.requireNonNull(cse, "cse cannot be null");
        Objects.requireNonNull(fuse, "fuse cannot be null");
        memory = memory == null ? MemoryConfig.defaults() : memory;

        List<OptimizerStage> normalized = new ArrayList<>(stageOrder.size());
        for (OptimizerStage stage : stageOrder) {
            if (stage == null) {
                throw new IllegalArgumentException("stageOrder cannot contain null");
            }
            normalized.add(stage);
        }
        LinkedHashSet<OptimizerStage> unique = new LinkedHashSet<>(normalized);
        if (unique.size() != normalized.size()) {
            throw new IllegalArgumentException("stageOrder cannot contain duplicates");
        }

        stageOrder = List.copyOf(normalized);
    }

    public static OptimizerConfig noOptimization() {
        return new OptimizerConfig(
                List.of(),
                CseConfig.strictDefaults(),
                FuseConfig.trainingDefaults(),
                MemoryConfig.defaults()
        );
    }

    public static OptimizerConfig trainingDefaults() {
        return new OptimizerConfig(
                List.of(OptimizerStage.AR, OptimizerStage.CSE, OptimizerStage.MEM),
                CseConfig.strictDefaults(),
                FuseConfig.trainingDefaults(),
                MemoryConfig.defaults()
        );
    }

    public static OptimizerConfig inferenceDefaults() {
        return new OptimizerConfig(
                List.of(OptimizerStage.AR, OptimizerStage.CSE, OptimizerStage.FUSE, OptimizerStage.MEM),
                CseConfig.aggressiveDefaults(),
                FuseConfig.inferenceDefaults(),
                MemoryConfig.defaults()
        );
    }

    public boolean isNoOptimization() {
        return stageOrder.isEmpty();
    }

    public boolean enables(OptimizerStage stage) {
        return stageOrder.contains(stage);
    }

    public OptimizerConfig withStageOrder(List<OptimizerStage> newStageOrder) {
        return new OptimizerConfig(newStageOrder, cse, fuse, memory);
    }

    public OptimizerConfig withCse(CseConfig newCse) {
        return new OptimizerConfig(stageOrder, newCse, fuse, memory);
    }

    public OptimizerConfig withFuse(FuseConfig newFuse) {
        return new OptimizerConfig(stageOrder, cse, newFuse, memory);
    }

    public OptimizerConfig withMemory(MemoryConfig newMemory) {
        return new OptimizerConfig(stageOrder, cse, fuse, newMemory);
    }
}
