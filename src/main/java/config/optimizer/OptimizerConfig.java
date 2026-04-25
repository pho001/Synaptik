package config.optimizer;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Objects;

public record OptimizerConfig(
        List<OptimizerStage> stageOrder,
        RewriteConfig rewrite,
        CseConfig cse,
        FuseConfig fuse,
        MemoryConfig memory,
        PartitionConfig partition
) {
    public OptimizerConfig {
        Objects.requireNonNull(stageOrder, "stageOrder cannot be null");
        rewrite = rewrite == null ? RewriteConfig.defaults() : rewrite;
        Objects.requireNonNull(cse, "cse cannot be null");
        Objects.requireNonNull(fuse, "fuse cannot be null");
        memory = memory == null ? MemoryConfig.defaults() : memory;
        partition = partition == null ? PartitionConfig.defaults() : partition;

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

    public OptimizerConfig(
            List<OptimizerStage> stageOrder,
            RewriteConfig rewrite,
            CseConfig cse,
            FuseConfig fuse,
            MemoryConfig memory
    ) {
        this(stageOrder, rewrite, cse, fuse, memory, PartitionConfig.defaults());
    }

    public OptimizerConfig(
            List<OptimizerStage> stageOrder,
            CseConfig cse,
            FuseConfig fuse,
            MemoryConfig memory
    ) {
        this(stageOrder, RewriteConfig.defaults(), cse, fuse, memory, PartitionConfig.defaults());
    }

    public static OptimizerConfig noOptimization() {
        return new OptimizerConfig(
                List.of(),
                RewriteConfig.defaults(),
                CseConfig.strictDefaults(),
                FuseConfig.trainingDefaults(),
                MemoryConfig.defaults(),
                PartitionConfig.defaults()
        );
    }

    public static OptimizerConfig trainingDefaults() {
        return new OptimizerConfig(
                List.of(OptimizerStage.AR, OptimizerStage.CSE, OptimizerStage.PART, OptimizerStage.MEM),
                RewriteConfig.defaults(),
                CseConfig.strictDefaults(),
                FuseConfig.trainingDefaults(),
                MemoryConfig.defaults(),
                PartitionConfig.defaults()
        );
    }

    public static OptimizerConfig inferenceDefaults() {
        return new OptimizerConfig(
                List.of(OptimizerStage.AR, OptimizerStage.CSE, OptimizerStage.PART, OptimizerStage.FUSE, OptimizerStage.MEM),
                RewriteConfig.defaults(),
                CseConfig.aggressiveDefaults(),
                FuseConfig.inferenceDefaults(),
                MemoryConfig.defaults(),
                PartitionConfig.defaults()
        );
    }

    public boolean isNoOptimization() {
        return stageOrder.isEmpty();
    }

    public boolean enables(OptimizerStage stage) {
        return stageOrder.contains(stage);
    }

    public OptimizerConfig withStageOrder(List<OptimizerStage> newStageOrder) {
        return new OptimizerConfig(newStageOrder, rewrite, cse, fuse, memory, partition);
    }

    public OptimizerConfig withRewrite(RewriteConfig newRewrite) {
        return new OptimizerConfig(stageOrder, newRewrite, cse, fuse, memory, partition);
    }

    public OptimizerConfig withCse(CseConfig newCse) {
        return new OptimizerConfig(stageOrder, rewrite, newCse, fuse, memory, partition);
    }

    public OptimizerConfig withFuse(FuseConfig newFuse) {
        return new OptimizerConfig(stageOrder, rewrite, cse, newFuse, memory, partition);
    }

    public OptimizerConfig withMemory(MemoryConfig newMemory) {
        return new OptimizerConfig(stageOrder, rewrite, cse, fuse, newMemory, partition);
    }

    public OptimizerConfig withPartition(PartitionConfig newPartition) {
        return new OptimizerConfig(stageOrder, rewrite, cse, fuse, memory, newPartition);
    }
}
