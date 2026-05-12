package config.compile;

import config.optimizer.CpuFusionConfig;
import config.optimizer.FuseConfig;

/**
 * Optimization policy inside already-owned execution regions.
 */
public record RegionOptimizationConfig(
        boolean enabled,
        FuseConfig fuse,
        CpuFusionConfig cpuFusion
) {
    public RegionOptimizationConfig {
        fuse = fuse == null ? FuseConfig.inferenceDefaults() : fuse;
        cpuFusion = cpuFusion == null ? CpuFusionConfig.defaults() : cpuFusion;
    }

    public static RegionOptimizationConfig trainingDefaults() {
        return new RegionOptimizationConfig(true, FuseConfig.trainingDefaults(), CpuFusionConfig.defaults());
    }

    public static RegionOptimizationConfig inferenceDefaults() {
        return new RegionOptimizationConfig(true, FuseConfig.inferenceDefaults(), CpuFusionConfig.defaults());
    }

    public static RegionOptimizationConfig disabled() {
        return new RegionOptimizationConfig(false, FuseConfig.inferenceDefaults(), CpuFusionConfig.off());
    }

    public RegionOptimizationConfig withCpuFusion(CpuFusionConfig newCpuFusion) {
        return new RegionOptimizationConfig(enabled, fuse, newCpuFusion);
    }

    public RegionOptimizationConfig withFuse(FuseConfig newFuse) {
        return new RegionOptimizationConfig(enabled, newFuse, cpuFusion);
    }
}
