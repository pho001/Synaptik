package backend.lowering.region;

public sealed interface RegionBackendPayload
        permits EmptyRegionPayload, CpuFusedRegionPayload, CpuSpecializedPrimitivePayload, MetalRegionPayload,
        CudaRegionPayload {
}
