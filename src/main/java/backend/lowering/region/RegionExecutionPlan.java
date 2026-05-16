package backend.lowering.region;

import backend.lowering.LoweredUnitArtifact;
import backend.lowering.LoweringFamily;
import graph.optimizer.partition.PartitionTarget;

import java.util.LinkedHashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

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
        if (regionId.isBlank()) {
            throw new IllegalArgumentException("regionId cannot be blank");
        }
        target = target == null ? PartitionTarget.NONE : target;
        loweringFamily = Objects.requireNonNull(loweringFamily, "loweringFamily cannot be null");
        if (anchorNodeId < 0) {
            throw new IllegalArgumentException("anchorNodeId must be >= 0");
        }
        orderedNodeIds = List.copyOf(orderedNodeIds == null ? List.of() : orderedNodeIds);
        requireUnique(orderedNodeIds, "orderedNodeIds");
        if (!orderedNodeIds.contains(anchorNodeId)) {
            throw new IllegalArgumentException("orderedNodeIds must contain anchorNodeId=" + anchorNodeId);
        }
        externalInputNodeIds = List.copyOf(externalInputNodeIds == null ? List.of() : externalInputNodeIds);
        boundaryOutputNodeIds = List.copyOf(boundaryOutputNodeIds == null ? List.of() : boundaryOutputNodeIds);
        requireUnique(boundaryOutputNodeIds, "boundaryOutputNodeIds");
        requireSubset(boundaryOutputNodeIds, orderedNodeIds, "boundaryOutputNodeIds");
        nodePlans = List.copyOf(nodePlans == null ? List.of() : nodePlans);
        executionGroups = List.copyOf(executionGroups == null ? List.of() : executionGroups);
        validateNodePlans(nodePlans, orderedNodeIds);
        validateExecutionGroups(executionGroups, orderedNodeIds);
        cost = cost == null ? RegionCost.ofWork(0L) : cost;
        decision = decision == null ? RegionDecision.selected(loweringFamily.id(), "selected") : decision;
        backendPayload = backendPayload == null ? EmptyRegionPayload.INSTANCE : backendPayload;
    }

    public boolean hasBoundaryOutput(int nodeId) {
        return boundaryOutputNodeIds.contains(nodeId);
    }

    private static void validateNodePlans(List<RegionNodePlan> nodePlans, List<Integer> orderedNodeIds) {
        LinkedHashSet<Integer> plannedNodeIds = new LinkedHashSet<>();
        for (RegionNodePlan nodePlan : nodePlans) {
            if (nodePlan == null) {
                throw new IllegalArgumentException("nodePlans cannot contain null");
            }
            if (!orderedNodeIds.contains(nodePlan.nodeId())) {
                throw new IllegalArgumentException("nodePlans contain node outside orderedNodeIds: " + nodePlan.nodeId());
            }
            if (!plannedNodeIds.add(nodePlan.nodeId())) {
                throw new IllegalArgumentException("nodePlans contain duplicate nodeId=" + nodePlan.nodeId());
            }
        }
    }

    private static void validateExecutionGroups(
            List<RegionExecutionGroup> executionGroups,
            List<Integer> orderedNodeIds
    ) {
        for (RegionExecutionGroup group : executionGroups) {
            if (group == null) {
                throw new IllegalArgumentException("executionGroups cannot contain null");
            }
            requireSubset(group.orderedNodeIds(), orderedNodeIds, "executionGroup.orderedNodeIds");
            requireSubset(group.outputNodeIds(), orderedNodeIds, "executionGroup.outputNodeIds");
        }
    }

    private static void requireUnique(List<Integer> nodeIds, String fieldName) {
        Set<Integer> unique = new LinkedHashSet<>(nodeIds);
        if (unique.size() != nodeIds.size()) {
            throw new IllegalArgumentException(fieldName + " cannot contain duplicates");
        }
    }

    private static void requireSubset(List<Integer> nodeIds, List<Integer> allowedNodeIds, String fieldName) {
        for (int nodeId : nodeIds) {
            if (!allowedNodeIds.contains(nodeId)) {
                throw new IllegalArgumentException(fieldName + " contains node outside orderedNodeIds: " + nodeId);
            }
        }
    }
}
