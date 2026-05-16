package backend.lowering.region;

import backend.accelerator.lowering.GpuCompoundLoweringArtifact;
import backend.accelerator.lowering.GpuLoweredRegionManifest;

public record MetalRegionPayload(
        GpuCompoundLoweringArtifact compoundArtifact,
        GpuLoweredRegionManifest manifest
) implements RegionBackendPayload {
}
