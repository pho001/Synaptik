package graph.optimizer.region.lowering;

import graph.CompiledNode;
import graph.optimizer.partition.PartitionTarget;
import graph.optimizer.region.RegionOptimizationContext;

import java.util.List;
import java.util.Objects;

/**
 * Context visible to region-aware lowering policy.
 */
public record RegionLoweringPolicyContext(
        PartitionTarget target,
        String regionId,
        List<Integer> orderedNodeIds,
        RegionOptimizationContext regionContext
) {
    public RegionLoweringPolicyContext {
        target = target == null ? PartitionTarget.NONE : target;
        regionId = regionId == null ? "" : regionId;
        orderedNodeIds = List.copyOf(orderedNodeIds == null ? List.of() : orderedNodeIds);
        regionContext = Objects.requireNonNull(regionContext, "regionContext cannot be null");
    }

    public CompiledNode compiledNode(int nodeId) {
        return regionContext.compiledNode(nodeId);
    }
}
