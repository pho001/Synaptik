package backend.accelerator.lowering;

import planning.partition.PartitionPlan;

/**
 * Backend-owned extension for accelerator plans that expose lowered-partition metadata.
 */
public interface AcceleratorPartitionPlan extends PartitionPlan {
    GpuLoweredPartitionManifest gpuLoweredPartitionManifest();
}
