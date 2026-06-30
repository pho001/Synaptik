package backend.accelerator.lowering;

import planning.partition.PartitionPlan;

/**
 * Backend-owned extension for accelerator plans that expose lowered-region metadata.
 */
public interface AcceleratorPartitionPlan extends PartitionPlan {
    GpuLoweredRegionManifest gpuLoweredRegionManifest();
}
