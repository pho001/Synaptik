package graph.execution.trace;

import graph.optimizer.partition.PartitionTarget;
import graph.optimizer.partition.PartitionPlannerStrategy;

import java.util.List;

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
