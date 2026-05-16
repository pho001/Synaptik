package backend.lowering.region;

import backend.lowering.LoweredUnitArtifact;
import backend.lowering.LoweringFamily;
import graph.optimizer.partition.PartitionTarget;

import java.util.List;
import java.util.Objects;

public record RegionExecutionPlan(
        String regionId,
        PartitionTarget target,
        LoweringFamily loweringFamily,
        int anchorNodeId,
        List<Integer> orderedNodeIds,
        List<Integer> externalInputNodeIds,
        List<Integer> boundaryOutputNodeIds,
        List<RegionNodePlan> nodePlans,
        List<RegionExecutionGroup> executionGroups,
        RegionCost cost,
        RegionDecision decision,
        RegionBackendPayload backendPayload
) implements LoweredUnitArtifact {
    public RegionExecutionPlan {
        regionId = regionId == null ? "" : regionId;
        target = target == null ? PartitionTarget.NONE : target;
        loweringFamily = Objects.requireNonNull(loweringFamily, "loweringFamily cannot be null");
        if (anchorNodeId < 0) {
            throw new IllegalArgumentException("anchorNodeId must be >= 0");
        }
        orderedNodeIds = List.copyOf(orderedNodeIds == null ? List.of() : orderedNodeIds);
        if (!orderedNodeIds.contains(anchorNodeId)) {
            throw new IllegalArgumentException("orderedNodeIds must contain anchorNodeId=" + anchorNodeId);
        }
        externalInputNodeIds = List.copyOf(externalInputNodeIds == null ? List.of() : externalInputNodeIds);
        boundaryOutputNodeIds = List.copyOf(boundaryOutputNodeIds == null ? List.of() : boundaryOutputNodeIds);
        nodePlans = List.copyOf(nodePlans == null ? List.of() : nodePlans);
        executionGroups = List.copyOf(executionGroups == null ? List.of() : executionGroups);
        cost = cost == null ? RegionCost.ofWork(0L) : cost;
        decision = decision == null ? RegionDecision.selected(loweringFamily.id(), "selected") : decision;
        backendPayload = backendPayload == null ? EmptyRegionPayload.INSTANCE : backendPayload;
    }

    public boolean hasBoundaryOutput(int nodeId) {
        return boundaryOutputNodeIds.contains(nodeId);
    }
}
