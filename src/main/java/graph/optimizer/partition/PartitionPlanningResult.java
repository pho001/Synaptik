package graph.optimizer.partition;

import graph.execution.trace.AcceleratorPartitionCompileTrace;

import java.util.List;

public record PartitionPlanningResult(
        List<AcceleratorPartitionPlan> plans,
        AcceleratorPartitionCompileTrace trace
) {
    public PartitionPlanningResult {
        plans = List.copyOf(plans == null ? List.of() : plans);
        trace = trace == null ? AcceleratorPartitionCompileTrace.empty() : trace;
    }

    public static PartitionPlanningResult empty() {
        return new PartitionPlanningResult(List.of(), AcceleratorPartitionCompileTrace.empty());
    }
}
