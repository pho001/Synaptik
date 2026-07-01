package backend.lowering.partition;

import backend.accelerator.lowering.GpuCompoundLoweringArtifact;
import backend.accelerator.lowering.GpuLoweredPartitionManifest;

public record MetalPartitionPayload(
        GpuCompoundLoweringArtifact compoundArtifact,
        GpuLoweredPartitionManifest manifest
) implements PartitionBackendPayload {
}
