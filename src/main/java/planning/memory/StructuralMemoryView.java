package planning.memory;

import planning.value.GraphValueRef;

import java.util.List;
import java.util.Objects;

/**
 * Structural view of partition value memory flow.
 *
 * @param plannedPartitionIds planned partition ids in order
 * @param materializedValues partition values that require storage
 * @param continuationValues partition values continued between execution units
 * @param virtualValues partition values represented without storage
 * @param valueFlows producer and consumer flow for partition values
 */
public record StructuralMemoryView(
        List<String> plannedPartitionIds,
        List<GraphValueRef> materializedValues,
        List<GraphValueRef> continuationValues,
        List<GraphValueRef> virtualValues,
        List<StructuralValueFlow> valueFlows
) {
    public StructuralMemoryView {
        plannedPartitionIds = List.copyOf(plannedPartitionIds == null ? List.of() : plannedPartitionIds);
        materializedValues = List.copyOf(materializedValues == null ? List.of() : materializedValues);
        continuationValues = List.copyOf(continuationValues == null ? List.of() : continuationValues);
        virtualValues = List.copyOf(virtualValues == null ? List.of() : virtualValues);
        valueFlows = List.copyOf(valueFlows == null ? List.of() : valueFlows);
    }

    /**
     * Creates a structural view without explicit value-flow records.
     *
     * @param plannedPartitionIds planned partition ids
     * @param materializedValues materialized values
     * @param continuationValues continuation values
     * @param virtualValues virtual values
     */
    public StructuralMemoryView(
            List<String> plannedPartitionIds,
            List<GraphValueRef> materializedValues,
            List<GraphValueRef> continuationValues,
            List<GraphValueRef> virtualValues
    ) {
        this(plannedPartitionIds, materializedValues, continuationValues, virtualValues, List.of());
    }

    /**
     * Finds flow metadata for a partition value.
     *
     * @param valueRef graph value reference
     * @return matching flow, or {@code null} when absent
     */
    public StructuralValueFlow flowOf(GraphValueRef valueRef) {
        Objects.requireNonNull(valueRef, "valueRef cannot be null");
        for (StructuralValueFlow flow : valueFlows) {
            if (flow.valueRef().equals(valueRef)) {
                return flow;
            }
        }
        return null;
    }

    /**
     * Counts values consumed by a different partition than their producer.
     *
     * @return cross-partition dependency count
     */
    public int crossPartitionDependencyCount() {
        int count = 0;
        for (StructuralValueFlow flow : valueFlows) {
            if (flow.hasCrossPartitionConsumer()) {
                count++;
            }
        }
        return count;
    }

    /**
     * Returns an empty structural memory view.
     *
     * @return empty view
     */
    public static StructuralMemoryView empty() {
        return new StructuralMemoryView(List.of(), List.of(), List.of(), List.of(), List.of());
    }
}
