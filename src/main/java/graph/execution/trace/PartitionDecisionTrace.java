package graph.execution.trace;

import graph.optimizer.partition.PartitionTarget;
import graph.optimizer.partition.PartitionPlannerStrategy;

import java.util.List;

public record PartitionDecisionTrace(
        PartitionPlannerStrategy strategy,
        PartitionTarget target,
        int startNodeId,
        boolean accepted,
        String reason,
        List<Integer> nodeIds,
        List<Integer> structuralNodeIds,
        List<String> opTypes,
        long estimatedWork,
        double selectedScore,
        double structuralScore,
        int exploredCandidates,
        boolean searchBudgetHit,
        int rejectedNodeId
) {
    public PartitionDecisionTrace {
        strategy = strategy == null ? PartitionPlannerStrategy.GREEDY_MAX_REGION : strategy;
        target = target == null ? PartitionTarget.NONE : target;
        reason = reason == null ? "" : reason;
        nodeIds = List.copyOf(nodeIds == null ? List.of() : nodeIds);
        structuralNodeIds = List.copyOf(structuralNodeIds == null ? nodeIds : structuralNodeIds);
        opTypes = List.copyOf(opTypes == null ? List.of() : opTypes);
        estimatedWork = Math.max(0L, estimatedWork);
        exploredCandidates = Math.max(0, exploredCandidates);
        rejectedNodeId = Math.max(-1, rejectedNodeId);
    }
}
