package graph.execution.trace;

import graph.optimizer.partition.PartitionTarget;
import graph.optimizer.partition.PartitionPlannerStrategy;

import java.util.List;

/**
 * Diagnostic decision for a partition candidate or seed.
 *
 * @param strategy planner strategy used
 * @param target partition target
 * @param startNodeId seed node id
 * @param accepted whether the candidate was accepted
 * @param reason diagnostic reason for acceptance or rejection
 * @param nodeIds node ids selected by the accepted candidate or attempted seed
 * @param structuralNodeIds node ids in the best structural candidate
 * @param opTypes operation names represented by the candidate
 * @param estimatedWork backend work estimate
 * @param selectedScore accepted candidate score
 * @param structuralScore best structural candidate score
 * @param exploredCandidates number of candidates explored
 * @param searchBudgetHit whether planner search limits were reached
 * @param rejectedNodeId node id that caused rejection, or {@code -1}
 */
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
