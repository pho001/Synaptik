package planning.partition.execution;

import planning.value.GraphValueRef;

import java.util.List;

/**
 * Executable unit and value-flow plan owned by one partition.
 *
 * <p>Partition identity, target, and kind deliberately live only on the owning partition. This record contains only
 * the execution details produced after backend ownership planning.</p>
 *
 * @param executionUnits execution units in partition order
 * @param executionValues values tracked across unit and partition boundaries
 * @param materializedOutputs outputs that must be materialized outside the partition
 * @param trace partition execution planning diagnostics
 */
public record PartitionExecutionPlan(
        List<ExecutionUnit> executionUnits,
        List<PartitionExecutionValue> executionValues,
        List<GraphValueRef> materializedOutputs,
        PartitionExecutionTrace trace
) {
    public PartitionExecutionPlan {
        executionUnits = List.copyOf(executionUnits == null ? List.of() : executionUnits);
        executionValues = List.copyOf(executionValues == null ? List.of() : executionValues);
        materializedOutputs = List.copyOf(materializedOutputs == null ? List.of() : materializedOutputs);
        trace = trace == null ? PartitionExecutionTrace.empty() : trace;
    }
}
