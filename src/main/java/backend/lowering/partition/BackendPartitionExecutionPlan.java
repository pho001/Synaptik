package backend.lowering.partition;

import backend.lowering.LoweredUnitArtifact;
import backend.lowering.LoweringFamily;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

public record BackendPartitionExecutionPlan(
        String executionPlanId,
        LoweringFamily loweringFamily,
        int anchorNodeId,
        List<Integer> orderedNodeIds,
        List<Integer> externalInputNodeIds,
        List<Integer> boundaryOutputNodeIds,
        List<PartitionNodePlan> nodePlans,
        List<PartitionExecutionGroup> executionGroups,
        PartitionCost cost,
        PartitionDecision decision,
        PartitionBackendPayload backendPayload
) implements LoweredUnitArtifact {
    public BackendPartitionExecutionPlan {
        executionPlanId = executionPlanId == null ? "" : executionPlanId;
        if (executionPlanId.isBlank()) {
            throw new IllegalArgumentException("executionPlanId cannot be blank");
        }
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
        cost = cost == null ? PartitionCost.ofWork(0L) : cost;
        decision = decision == null ? PartitionDecision.selected(loweringFamily.id(), "selected") : decision;
        backendPayload = backendPayload == null ? EmptyPartitionPayload.INSTANCE : backendPayload;
    }

    public boolean hasBoundaryOutput(int nodeId) {
        return boundaryOutputNodeIds.contains(nodeId);
    }

    private static void validateNodePlans(List<PartitionNodePlan> nodePlans, List<Integer> orderedNodeIds) {
        LinkedHashSet<Integer> plannedNodeIds = new LinkedHashSet<>();
        for (PartitionNodePlan nodePlan : nodePlans) {
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
            List<PartitionExecutionGroup> executionGroups,
            List<Integer> orderedNodeIds
    ) {
        for (PartitionExecutionGroup group : executionGroups) {
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
