package graph.optimizer.region;

import graph.optimizer.partition.PartitionValueRef;
import tensor.Tensor;

public record RegionValue(
        RegionValueRef ref,
        PartitionValueRef sourceValueRef,
        Tensor semanticTensor,
        int producerNodeId,
        int elementCount,
        ValueTransportKind transportKind,
        ValueTypeContract typeContract,
        boolean requiredMaterialized
) {
    public RegionValue {
        if (ref == null || sourceValueRef == null || semanticTensor == null || transportKind == null || typeContract == null) {
            throw new IllegalArgumentException("RegionValue fields cannot be null");
        }
        if (producerNodeId < 0 || elementCount < 0) {
            throw new IllegalArgumentException("producerNodeId and elementCount must be >= 0");
        }
    }
}
