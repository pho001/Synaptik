package trace.compile;

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
        String strategy,
        String target,
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
        MaterializationCostTrace costSummary,
        List<CandidateCostTrace> finalists
) {
    public PartitionDecisionTrace(
            String strategy,
            String target,
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
        strategy = strategy == null ? "GREEDY_MAX_PARTITION" : strategy;
        target = target == null ? "NONE" : target;
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

    public static PartitionDecisionTrace coveredByEarlierPartition(
            String strategy,
            String target,
            int nodeId,
            List<String> opTypes
    ) {
        return new PartitionDecisionTrace(
                strategy,
                target,
                nodeId,
                false,
                "covered-by-earlier-partition",
                List.of(nodeId),
                List.of(nodeId),
                opTypes,
                0L,
                Double.NEGATIVE_INFINITY,
                Double.NEGATIVE_INFINITY,
                0,
                false,
                -1
        );
    }

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
