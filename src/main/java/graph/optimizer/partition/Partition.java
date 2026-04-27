package graph.optimizer.partition;

import graph.execution.trace.PartitionDecisionTrace;
import graph.optimizer.partition.cost.AcceleratorPartitionScoreModel;

import java.util.List;

public record Partition(
        String partitionId,
        PartitionTarget target,
        List<Integer> orderedNodeIds,
        List<PartitionValue> values,
        List<PartitionEdge> internalEdges,
        List<Integer> externalInputNodeIds,
        List<PartitionValueRef> outputValueRefs,
        int anchorSeedNodeId,
        List<PartitionValueRef> requiredMaterializedValueRefs,
        List<PartitionEdge> boundaryEdges,
        List<PartitionBoundaryReason> boundaryReasons,
        long estimatedWork,
        AcceleratorPartitionScoreModel.CandidateMetrics structuralMetrics,
        PartitionPlannerStrategy plannerStrategy,
        PartitionDecisionTrace debugTrace
) {
    public Partition {
        partitionId = partitionId == null ? "" : partitionId;
        target = target == null ? PartitionTarget.NONE : target;
        orderedNodeIds = List.copyOf(orderedNodeIds == null ? List.of() : orderedNodeIds);
        values = List.copyOf(values == null ? List.of() : values);
        internalEdges = List.copyOf(internalEdges == null ? List.of() : internalEdges);
        externalInputNodeIds = List.copyOf(externalInputNodeIds == null ? List.of() : externalInputNodeIds);
        outputValueRefs = List.copyOf(outputValueRefs == null ? List.of() : outputValueRefs);
        requiredMaterializedValueRefs = List.copyOf(requiredMaterializedValueRefs == null ? List.of() : requiredMaterializedValueRefs);
        boundaryEdges = List.copyOf(boundaryEdges == null ? List.of() : boundaryEdges);
        boundaryReasons = List.copyOf(boundaryReasons == null ? List.of() : boundaryReasons);
        estimatedWork = Math.max(0L, estimatedWork);
        structuralMetrics = structuralMetrics == null
                ? new AcceleratorPartitionScoreModel.CandidateMetrics(0, 0, 0, 0, 0)
                : structuralMetrics;
        plannerStrategy = plannerStrategy == null ? PartitionPlannerStrategy.GREEDY_MAX_REGION : plannerStrategy;
        if (anchorSeedNodeId < 0) {
            throw new IllegalArgumentException("anchorSeedNodeId must be >= 0");
        }
    }
}
