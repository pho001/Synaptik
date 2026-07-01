package planning.partition;

import planning.partition.execution.PartitionExecutionPlan;

import java.util.Objects;

/**
 * Immutable enrichment of an accepted backend partition with its execution plan.
 *
 * <p>Backend ownership planning produces {@link PlannedPartition}. Partition execution planning then creates this
 * complete compile-to-lowering contract, avoiding both nullable partial state and parallel partition/plan lists.</p>
 *
 * @param plannedPartition accepted partition and backend plan
 * @param executionPlan execution units and value-flow metadata owned by that partition
 */
public record ExecutablePartitionPlan(
        PlannedPartition plannedPartition,
        PartitionExecutionPlan executionPlan
) {
    public ExecutablePartitionPlan {
        plannedPartition = Objects.requireNonNull(plannedPartition, "plannedPartition cannot be null");
        executionPlan = Objects.requireNonNull(executionPlan, "executionPlan cannot be null");
    }

    public Partition partition() {
        return plannedPartition.partition();
    }

    public PartitionPlan backendPlan() {
        return plannedPartition.plan();
    }
}
