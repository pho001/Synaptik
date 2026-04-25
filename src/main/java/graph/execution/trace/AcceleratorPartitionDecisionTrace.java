package graph.execution.trace;

import graph.optimizer.partition.AcceleratorTarget;
import graph.optimizer.partition.PartitionPlannerStrategy;

import java.util.List;

public record AcceleratorPartitionDecisionTrace(
        PartitionPlannerStrategy strategy,
        AcceleratorTarget target,
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
    public AcceleratorPartitionDecisionTrace {
        strategy = strategy == null ? PartitionPlannerStrategy.GREEDY_MAX_REGION : strategy;
        target = target == null ? AcceleratorTarget.NONE : target;
        reason = reason == null ? "" : reason;
        nodeIds = List.copyOf(nodeIds == null ? List.of() : nodeIds);
        structuralNodeIds = List.copyOf(structuralNodeIds == null ? nodeIds : structuralNodeIds);
        opTypes = List.copyOf(opTypes == null ? List.of() : opTypes);
        estimatedWork = Math.max(0L, estimatedWork);
        exploredCandidates = Math.max(0, exploredCandidates);
        rejectedNodeId = Math.max(-1, rejectedNodeId);
    }
}
