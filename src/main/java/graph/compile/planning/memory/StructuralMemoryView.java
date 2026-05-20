package graph.compile.planning.memory;

import graph.compile.planning.value.GraphValueRef;

import java.util.List;
import java.util.Objects;

/**
 * Structural view of region value memory flow.
 *
 * @param optimizedRegionIds optimized region ids in order
 * @param materializedValues region values that require storage
 * @param continuationValues region values continued between execution units
 * @param virtualValues region values represented without storage
 * @param valueFlows producer and consumer flow for region values
 */
public record StructuralMemoryView(
        List<String> optimizedRegionIds,
        List<GraphValueRef> materializedValues,
        List<GraphValueRef> continuationValues,
        List<GraphValueRef> virtualValues,
        List<StructuralValueFlow> valueFlows
) {
    public StructuralMemoryView {
        optimizedRegionIds = List.copyOf(optimizedRegionIds == null ? List.of() : optimizedRegionIds);
        materializedValues = List.copyOf(materializedValues == null ? List.of() : materializedValues);
        continuationValues = List.copyOf(continuationValues == null ? List.of() : continuationValues);
        virtualValues = List.copyOf(virtualValues == null ? List.of() : virtualValues);
        valueFlows = List.copyOf(valueFlows == null ? List.of() : valueFlows);
    }

    /**
     * Creates a structural view without explicit value-flow records.
     *
     * @param optimizedRegionIds optimized region ids
     * @param materializedValues materialized values
     * @param continuationValues continuation values
     * @param virtualValues virtual values
     */
    public StructuralMemoryView(
            List<String> optimizedRegionIds,
            List<GraphValueRef> materializedValues,
            List<GraphValueRef> continuationValues,
            List<GraphValueRef> virtualValues
    ) {
        this(optimizedRegionIds, materializedValues, continuationValues, virtualValues, List.of());
    }

    /**
     * Finds flow metadata for a region value.
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
     * Counts values consumed by a different region than their producer.
     *
     * @return cross-region dependency count
     */
    public int crossRegionDependencyCount() {
        int count = 0;
        for (StructuralValueFlow flow : valueFlows) {
            if (flow.hasCrossRegionConsumer()) {
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
