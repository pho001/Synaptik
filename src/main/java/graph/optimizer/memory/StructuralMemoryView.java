package graph.optimizer.memory;

import graph.optimizer.region.RegionValueRef;

import java.util.List;
import java.util.Objects;

public record StructuralMemoryView(
        List<String> optimizedRegionIds,
        List<RegionValueRef> materializedValues,
        List<RegionValueRef> continuationValues,
        List<RegionValueRef> virtualValues,
        List<StructuralValueFlow> valueFlows
) {
    public StructuralMemoryView {
        optimizedRegionIds = List.copyOf(optimizedRegionIds == null ? List.of() : optimizedRegionIds);
        materializedValues = List.copyOf(materializedValues == null ? List.of() : materializedValues);
        continuationValues = List.copyOf(continuationValues == null ? List.of() : continuationValues);
        virtualValues = List.copyOf(virtualValues == null ? List.of() : virtualValues);
        valueFlows = List.copyOf(valueFlows == null ? List.of() : valueFlows);
    }

    public StructuralMemoryView(
            List<String> optimizedRegionIds,
            List<RegionValueRef> materializedValues,
            List<RegionValueRef> continuationValues,
            List<RegionValueRef> virtualValues
    ) {
        this(optimizedRegionIds, materializedValues, continuationValues, virtualValues, List.of());
    }

    public StructuralValueFlow flowOf(RegionValueRef valueRef) {
        Objects.requireNonNull(valueRef, "valueRef cannot be null");
        for (StructuralValueFlow flow : valueFlows) {
            if (flow.valueRef().equals(valueRef)) {
                return flow;
            }
        }
        return null;
    }

    public int crossRegionDependencyCount() {
        int count = 0;
        for (StructuralValueFlow flow : valueFlows) {
            if (flow.hasCrossRegionConsumer()) {
                count++;
            }
        }
        return count;
    }

    public static StructuralMemoryView empty() {
        return new StructuralMemoryView(List.of(), List.of(), List.of(), List.of(), List.of());
    }
}
