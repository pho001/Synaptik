package graph.execution.trace;

import graph.optimizer.partition.PartitionTarget;
import graph.optimizer.partition.PartitionPlannerStrategy;
import graph.optimizer.partition.cost.AcceleratorPartitionScoreModel;

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
 * @param costSummary static materialization-aware score summary, if available
 * @param finalists bounded rejected finalist summaries
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
        int rejectedNodeId,
        AcceleratorPartitionScoreModel.MaterializationCostSummary costSummary,
        List<CandidateCostTrace> finalists
) {
    public PartitionDecisionTrace(
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
        this(
                strategy,
                target,
                startNodeId,
                accepted,
                reason,
                nodeIds,
                structuralNodeIds,
                opTypes,
                estimatedWork,
                selectedScore,
                structuralScore,
                exploredCandidates,
                searchBudgetHit,
                rejectedNodeId,
                null,
                List.of()
        );
    }

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
        finalists = List.copyOf(finalists == null ? List.of() : finalists).stream()
                .limit(3)
                .toList();
    }

    /**
     * Accessor note: callers use {@link #finalists()} for the bounded finalist list.
     */

    /**
     * Compact finalist score summary for rejected or non-winning candidates.
     *
     * @param nodeIds candidate node ids
     * @param reason finalist rejection reason
     * @param finalScore materialization-aware final score
     * @param boundaryCount CPU/accelerator boundary count
     * @param estimatedTransferBytes upload plus download bytes
     * @param layoutFallbackBytes bytes affected by layout fallback or GPU-side dense materialization
     * @param estimatedComputeWork backend work estimate
     * @param preset selected static preset
     */
    public record CandidateCostTrace(
            List<Integer> nodeIds,
            String reason,
            double finalScore,
            int boundaryCount,
            long estimatedTransferBytes,
            long layoutFallbackBytes,
            long estimatedComputeWork,
            String preset
    ) {
        public CandidateCostTrace {
            nodeIds = List.copyOf(nodeIds == null ? List.of() : nodeIds);
            reason = reason == null ? "" : reason;
            boundaryCount = Math.max(0, boundaryCount);
            estimatedTransferBytes = Math.max(0L, estimatedTransferBytes);
            layoutFallbackBytes = Math.max(0L, layoutFallbackBytes);
            estimatedComputeWork = Math.max(0L, estimatedComputeWork);
            preset = preset == null ? "" : preset;
        }
    }
}
