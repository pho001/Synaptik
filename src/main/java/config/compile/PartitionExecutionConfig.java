package config.compile;

import config.optimizer.CpuFusionConfig;
import config.optimizer.FuseConfig;

/**
 * Optimization policy inside already-owned execution partitions.
 */
public record PartitionExecutionConfig(
        boolean enabled,
        FuseConfig fuse,
        CpuFusionConfig cpuFusion
) {
    public PartitionExecutionConfig {
        fuse = fuse == null ? FuseConfig.inferenceDefaults() : fuse;
        cpuFusion = cpuFusion == null ? CpuFusionConfig.defaults() : cpuFusion;
    }

    public static PartitionExecutionConfig trainingDefaults() {
        return new PartitionExecutionConfig(true, FuseConfig.trainingDefaults(), CpuFusionConfig.defaults());
    }

    public static PartitionExecutionConfig inferenceDefaults() {
        return new PartitionExecutionConfig(true, FuseConfig.inferenceDefaults(), CpuFusionConfig.defaults());
    }

    public static PartitionExecutionConfig disabled() {
        return new PartitionExecutionConfig(false, FuseConfig.inferenceDefaults(), CpuFusionConfig.off());
    }

    public PartitionExecutionConfig withCpuFusion(CpuFusionConfig newCpuFusion) {
        return new PartitionExecutionConfig(enabled, fuse, newCpuFusion);
    }

    public PartitionExecutionConfig withFuse(FuseConfig newFuse) {
        return new PartitionExecutionConfig(enabled, newFuse, cpuFusion);
    }
}
