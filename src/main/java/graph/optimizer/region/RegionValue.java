package graph.optimizer.region;

import graph.optimizer.partition.PartitionValueRef;
import tensor.Tensor;

/**
 * Value tracked inside an optimized region.
 *
 * @param ref region-scoped value reference
 * @param sourceValueRef original partition value reference
 * @param semanticTensor tensor represented by this region value
 * @param producerNodeId compiled node that produces the value
 * @param elementCount value size in elements
 * @param transportKind whether the value is materialized, virtual, or continued between units
 * @param typeContract logical/storage/compute/transport type contract
 * @param requiredMaterialized whether graph semantics require materialized storage at the region boundary
 */
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
