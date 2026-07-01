package backend.lowering.partition;

public sealed interface PartitionBackendPayload
        permits EmptyPartitionPayload, CpuFusedPartitionPayload, CpuSpecializedPrimitivePayload, MetalPartitionPayload,
        CudaPartitionPayload {
}
