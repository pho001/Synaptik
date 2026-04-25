package graph.execution.trace;

import graph.optimizer.partition.AcceleratorTarget;
import graph.optimizer.partition.PartitionPlannerStrategy;

import java.util.List;

public record AcceleratorPartitionCompileTrace(
        PartitionPlannerStrategy strategy,
        AcceleratorTarget target,
        int totalConsidered,
        int acceptedCount,
        int rejectedCount,
        List<AcceleratorPartitionDecisionTrace> decisions
) {
    public AcceleratorPartitionCompileTrace {
        strategy = strategy == null ? PartitionPlannerStrategy.GREEDY_MAX_REGION : strategy;
        target = target == null ? AcceleratorTarget.NONE : target;
        totalConsidered = Math.max(0, totalConsidered);
        acceptedCount = Math.max(0, acceptedCount);
        rejectedCount = Math.max(0, rejectedCount);
        decisions = List.copyOf(decisions == null ? List.of() : decisions);
    }

    public static AcceleratorPartitionCompileTrace empty() {
        return new AcceleratorPartitionCompileTrace(
                PartitionPlannerStrategy.GREEDY_MAX_REGION,
                AcceleratorTarget.NONE,
                0,
                0,
                0,
                List.of()
        );
    }
}
