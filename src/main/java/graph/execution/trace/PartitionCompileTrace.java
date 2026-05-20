package graph.execution.trace;

import graph.compile.planning.partition.PartitionTarget;
import graph.compile.planning.partition.PartitionPlannerStrategy;

import java.util.List;

/**
 * Partition planning diagnostics captured during compilation.
 *
 * @param strategy planner strategy used
 * @param target partition target
 * @param totalConsidered number of candidate starts considered
 * @param acceptedCount number of accepted partitions
 * @param rejectedCount number of rejected candidates
 * @param decisions detailed partition decisions
 */
public record PartitionCompileTrace(
        PartitionPlannerStrategy strategy,
        PartitionTarget target,
        int totalConsidered,
        int acceptedCount,
        int rejectedCount,
        List<PartitionDecisionTrace> decisions
) {
    public PartitionCompileTrace {
        strategy = strategy == null ? PartitionPlannerStrategy.GREEDY_MAX_REGION : strategy;
        target = target == null ? PartitionTarget.NONE : target;
        totalConsidered = Math.max(0, totalConsidered);
        acceptedCount = Math.max(0, acceptedCount);
        rejectedCount = Math.max(0, rejectedCount);
        decisions = List.copyOf(decisions == null ? List.of() : decisions);
    }

    /**
     * Returns an empty partition compile trace.
     *
     * @return empty trace
     */
    public static PartitionCompileTrace empty() {
        return new PartitionCompileTrace(
                PartitionPlannerStrategy.GREEDY_MAX_REGION,
                PartitionTarget.NONE,
                0,
                0,
                0,
                List.of()
        );
    }
}
