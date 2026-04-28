package config.optimizer;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Objects;

/**
 * Immutable configuration for the graph optimizer pipeline.
 *
 * <p>The optimizer is stage-based. {@code stageOrder} selects which stages run and in which order,
 * while the remaining components configure each stage. The normal training/inference order is
 * {@code AR -> CSE -> PART -> FUSE -> MEM}: algebraic/lowering rewrites, common subexpression
 * elimination, partition planning, region fusion, and memory planning.</p>
 *
 * <p>Construction validates the structural dependencies between stages. For example, {@code FUSE}
 * requires {@code PART}, and {@code MEM} requires {@code FUSE}, because memory planning depends on
 * the final fused/partitioned execution units.</p>
 *
 * @param stageOrder ordered optimizer stages; must not contain {@code null} or duplicates
 * @param rewrite algebraic and semantic lowering rewrite configuration
 * @param cse common subexpression elimination configuration
 * @param fuse region fusion configuration
 * @param memory memory planning configuration
 * @param partition shared partition/search limits
 * @param offload accelerator/offload policy
 * @param cpuRegion CPU execution region policy
 * @param cpuFusion CPU fused-loop policy
 */
public record OptimizerConfig(
        List<OptimizerStage> stageOrder,
        RewriteConfig rewrite,
        CseConfig cse,
        FuseConfig fuse,
        MemoryConfig memory,
        PartitionConfig partition,
        OffloadConfig offload,
        CpuRegionConfig cpuRegion,
        CpuFusionConfig cpuFusion
) {
    public OptimizerConfig {
        Objects.requireNonNull(stageOrder, "stageOrder cannot be null");
        rewrite = rewrite == null ? RewriteConfig.defaults() : rewrite;
        Objects.requireNonNull(cse, "cse cannot be null");
        Objects.requireNonNull(fuse, "fuse cannot be null");
        memory = memory == null ? MemoryConfig.defaults() : memory;
        partition = partition == null ? PartitionConfig.defaults() : partition;
        offload = offload == null ? OffloadConfig.defaults() : offload;
        cpuRegion = cpuRegion == null ? CpuRegionConfig.defaults() : cpuRegion;
        cpuFusion = cpuFusion == null ? CpuFusionConfig.defaults() : cpuFusion;

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
        validateStageOrdering(normalized);

        stageOrder = List.copyOf(normalized);
    }

    /**
     * Creates an optimizer config with default graph region/fusion policies.
     *
     * @param stageOrder ordered optimizer stages
     * @param rewrite rewrite configuration; {@code null} uses defaults
     * @param cse CSE configuration
     * @param fuse fusion configuration
     * @param memory memory planning configuration; {@code null} uses defaults
     * @param partition partition planning configuration; {@code null} uses defaults
     */
    public OptimizerConfig(
            List<OptimizerStage> stageOrder,
            RewriteConfig rewrite,
            CseConfig cse,
            FuseConfig fuse,
            MemoryConfig memory,
            PartitionConfig partition
    ) {
        this(
                stageOrder,
                rewrite,
                cse,
                fuse,
                memory,
                partition,
                OffloadConfig.defaults(),
                CpuRegionConfig.defaults(),
                CpuFusionConfig.defaults()
        );
    }

    /**
     * Creates an optimizer config with default partition planning.
     *
     * @param stageOrder ordered optimizer stages
     * @param rewrite rewrite configuration; {@code null} uses defaults
     * @param cse CSE configuration
     * @param fuse fusion configuration
     * @param memory memory planning configuration; {@code null} uses defaults
     */
    public OptimizerConfig(
            List<OptimizerStage> stageOrder,
            RewriteConfig rewrite,
            CseConfig cse,
            FuseConfig fuse,
            MemoryConfig memory
    ) {
        this(stageOrder, rewrite, cse, fuse, memory, PartitionConfig.defaults());
    }

    /**
     * Creates an optimizer config with default rewrite and partition settings.
     *
     * @param stageOrder ordered optimizer stages
     * @param cse CSE configuration
     * @param fuse fusion configuration
     * @param memory memory planning configuration
     */
    public OptimizerConfig(
            List<OptimizerStage> stageOrder,
            CseConfig cse,
            FuseConfig fuse,
            MemoryConfig memory
    ) {
        this(stageOrder, RewriteConfig.defaults(), cse, fuse, memory, PartitionConfig.defaults());
    }

    /**
     * Returns a profile with no optimizer stages enabled.
     *
     * <p>This is a graph-level baseline. Runtime acceleration such as vectorization or BLAS is controlled
     * separately by {@link config.runtime.RuntimeConfig}.</p>
     *
     * @return optimizer config whose {@code stageOrder} is empty
     */
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

    /**
     * Returns the default optimizer pipeline for training-capable graphs.
     *
     * @return defaults using strict CSE and training fusion/runtime assumptions
     */
    public static OptimizerConfig trainingDefaults() {
        return new OptimizerConfig(
                List.of(OptimizerStage.AR, OptimizerStage.CSE, OptimizerStage.PART, OptimizerStage.FUSE, OptimizerStage.MEM),
                RewriteConfig.defaults(),
                CseConfig.strictDefaults(),
                FuseConfig.trainingDefaults(),
                MemoryConfig.defaults(),
                PartitionConfig.defaults()
        );
    }

    /**
     * Returns the default optimizer pipeline for forward-only inference graphs.
     *
     * @return defaults using inference-oriented CSE and fusion settings
     */
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

    /**
     * Reports whether this config disables all optimizer stages.
     *
     * @return {@code true} when {@code stageOrder} is empty
     */
    public boolean isNoOptimization() {
        return stageOrder.isEmpty();
    }

    /**
     * Checks whether a stage is enabled in this pipeline.
     *
     * @param stage optimizer stage to test; {@code null} returns {@code false}
     * @return {@code true} when {@code stageOrder} contains {@code stage}
     */
    public boolean enables(OptimizerStage stage) {
        return stageOrder.contains(stage);
    }

    /**
     * Returns a copy with a different stage order and the same per-stage settings.
     *
     * @param newStageOrder replacement stage order
     * @return new optimizer config
     */
    public OptimizerConfig withStageOrder(List<OptimizerStage> newStageOrder) {
        return new OptimizerConfig(newStageOrder, rewrite, cse, fuse, memory, partition, offload, cpuRegion, cpuFusion);
    }

    /**
     * Returns a copy with different rewrite settings.
     *
     * @param newRewrite replacement rewrite config; {@code null} uses rewrite defaults
     * @return new optimizer config
     */
    public OptimizerConfig withRewrite(RewriteConfig newRewrite) {
        return new OptimizerConfig(stageOrder, newRewrite, cse, fuse, memory, partition, offload, cpuRegion, cpuFusion);
    }

    /**
     * Returns a copy with different CSE settings.
     *
     * @param newCse replacement CSE config; must not be {@code null}
     * @return new optimizer config
     */
    public OptimizerConfig withCse(CseConfig newCse) {
        return new OptimizerConfig(stageOrder, rewrite, newCse, fuse, memory, partition, offload, cpuRegion, cpuFusion);
    }

    /**
     * Returns a copy with different fusion settings.
     *
     * @param newFuse replacement fusion config; must not be {@code null}
     * @return new optimizer config
     */
    public OptimizerConfig withFuse(FuseConfig newFuse) {
        return new OptimizerConfig(stageOrder, rewrite, cse, newFuse, memory, partition, offload, cpuRegion, cpuFusion);
    }

    /**
     * Returns a copy with different memory planning settings.
     *
     * @param newMemory replacement memory config; {@code null} uses memory defaults
     * @return new optimizer config
     */
    public OptimizerConfig withMemory(MemoryConfig newMemory) {
        return new OptimizerConfig(stageOrder, rewrite, cse, fuse, newMemory, partition, offload, cpuRegion, cpuFusion);
    }

    /**
     * Returns a copy with different partition planning settings.
     *
     * @param newPartition replacement partition config; {@code null} uses partition defaults
     * @return new optimizer config
     */
    public OptimizerConfig withPartition(PartitionConfig newPartition) {
        return new OptimizerConfig(stageOrder, rewrite, cse, fuse, memory, newPartition, offload, cpuRegion, cpuFusion);
    }

    /**
     * Returns a copy with different offload policy.
     *
     * @param newOffload replacement offload config; {@code null} uses defaults
     * @return new optimizer config
     */
    public OptimizerConfig withOffload(OffloadConfig newOffload) {
        return new OptimizerConfig(stageOrder, rewrite, cse, fuse, memory, partition, newOffload, cpuRegion, cpuFusion);
    }

    /**
     * Returns a copy with different CPU region policy.
     *
     * @param newCpuRegion replacement CPU region config; {@code null} uses defaults
     * @return new optimizer config
     */
    public OptimizerConfig withCpuRegion(CpuRegionConfig newCpuRegion) {
        return new OptimizerConfig(stageOrder, rewrite, cse, fuse, memory, partition, offload, newCpuRegion, cpuFusion);
    }

    /**
     * Returns a copy with different CPU fusion policy.
     *
     * @param newCpuFusion replacement CPU fusion config; {@code null} uses defaults
     * @return new optimizer config
     */
    public OptimizerConfig withCpuFusion(CpuFusionConfig newCpuFusion) {
        return new OptimizerConfig(stageOrder, rewrite, cse, fuse, memory, partition, offload, cpuRegion, newCpuFusion);
    }

    private static void validateStageOrdering(List<OptimizerStage> stageOrder) {
        int partitionIndex = stageOrder.indexOf(OptimizerStage.PART);
        int fuseIndex = stageOrder.indexOf(OptimizerStage.FUSE);
        int memoryIndex = stageOrder.indexOf(OptimizerStage.MEM);
        if (fuseIndex >= 0 && partitionIndex < 0) {
            throw new IllegalArgumentException("Optimizer stage order is invalid: FUSE requires PART.");
        }
        if (partitionIndex >= 0 && fuseIndex >= 0 && partitionIndex > fuseIndex) {
            throw new IllegalArgumentException("Optimizer stage order is invalid: PART must run before FUSE.");
        }
        if (memoryIndex >= 0 && fuseIndex < 0) {
            throw new IllegalArgumentException("Optimizer stage order is invalid: MEM requires FUSE.");
        }
    }
}
