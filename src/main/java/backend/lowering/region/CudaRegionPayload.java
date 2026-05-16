package backend.lowering.region;

import backend.accelerator.lowering.GpuCompoundLoweringArtifact;
import backend.accelerator.lowering.GpuLoweredRegionManifest;

public record CudaRegionPayload(
        GpuCompoundLoweringArtifact compoundArtifact,
        GpuLoweredRegionManifest manifest
) implements RegionBackendPayload {
}
