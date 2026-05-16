package backend.lowering.region;

public sealed interface RegionBackendPayload
        permits EmptyRegionPayload, CpuFusedRegionPayload, CpuNativeRegionPayload, MetalRegionPayload, CudaRegionPayload {
}
