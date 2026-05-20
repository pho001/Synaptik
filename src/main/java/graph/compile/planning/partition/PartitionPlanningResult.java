package graph.compile.planning.partition;

import graph.execution.trace.PartitionCompileTrace;

import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Result of partition planning.
 *
 * @param partitions accepted partitions in graph order
 * @param plansByPartitionId backend plans keyed by partition id
 * @param trace planning diagnostics for compile tracing
 */
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

    /**
     * Returns an empty planning result.
     *
     * @return empty result with empty trace
     */
    public static PartitionPlanningResult empty() {
        return new PartitionPlanningResult(List.of(), Map.of(), PartitionCompileTrace.empty());
    }

    /**
     * Returns all non-null backend plans attached to partitions.
     *
     * @return attached plans
     */
    public List<PartitionPlan> attachedPlans() {
        return plansByPartitionId.values().stream()
                .filter(Objects::nonNull)
                .toList();
    }

    /**
     * Looks up a backend plan by partition id.
     *
     * @param partitionId partition id
     * @return plan, or {@code null} when no plan is attached
     */
    public PartitionPlan planForPartition(String partitionId) {
        return partitionId == null ? null : plansByPartitionId.get(partitionId);
    }
}
