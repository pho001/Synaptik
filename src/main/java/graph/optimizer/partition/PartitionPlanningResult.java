package graph.optimizer.partition;

import graph.execution.trace.PartitionCompileTrace;

import java.util.List;
import java.util.Map;
import java.util.Objects;

public record PartitionPlanningResult(
        List<Partition> partitions,
        Map<String, PartitionPlan> plansByPartitionId,
        PartitionCompileTrace trace
) {
    public PartitionPlanningResult {
        partitions = List.copyOf(partitions == null ? List.of() : partitions);
        plansByPartitionId = Map.copyOf(plansByPartitionId == null ? Map.of() : plansByPartitionId);
        trace = trace == null ? PartitionCompileTrace.empty() : trace;
    }

    public static PartitionPlanningResult empty() {
        return new PartitionPlanningResult(List.of(), Map.of(), PartitionCompileTrace.empty());
    }

    public List<PartitionPlan> attachedPlans() {
        return plansByPartitionId.values().stream()
                .filter(Objects::nonNull)
                .toList();
    }

    public PartitionPlan planForPartition(String partitionId) {
        return partitionId == null ? null : plansByPartitionId.get(partitionId);
    }
}
