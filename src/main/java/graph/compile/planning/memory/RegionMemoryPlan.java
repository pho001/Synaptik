package graph.compile.planning.memory;

import graph.compile.planning.value.GraphValueRef;
import tensor.Tensor;

import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.TreeMap;

public record RegionMemoryPlan(
        StructuralMemoryView structuralView,
        Map<GraphValueRef, RegionValueLifetime> valueLifetimes,
        Map<GraphValueRef, MaterializationPlanEntry> materializationPlan,
        Map<GraphValueRef, RegionMemoryBinding> memoryBindings,
        Map<GraphValueRef, Integer> slotByValueRef,
        Map<Integer, Integer> slotUseCounts,
        Map<Integer, Integer> slotSizes,
        Map<Tensor, GraphValueRef> tensorToGraphValueRef,
        Map<Integer, GraphValueRef> nodeIdToGraphValueRef,
        List<RegionHandoffRequirement> handoffRequirements
) {
    public RegionMemoryPlan(
            StructuralMemoryView structuralView,
            Map<GraphValueRef, RegionValueLifetime> valueLifetimes,
            Map<GraphValueRef, MaterializationPlanEntry> materializationPlan,
            Map<GraphValueRef, RegionMemoryBinding> memoryBindings,
            Map<GraphValueRef, Integer> slotByValueRef,
            Map<Integer, Integer> slotSizes,
            Map<Tensor, GraphValueRef> tensorToGraphValueRef,
            Map<Integer, GraphValueRef> nodeIdToGraphValueRef,
            List<RegionHandoffRequirement> handoffRequirements
    ) {
        this(
                structuralView,
                valueLifetimes,
                materializationPlan,
                memoryBindings,
                slotByValueRef,
                buildSlotUseCounts(slotByValueRef == null ? Map.of() : slotByValueRef),
                slotSizes,
                tensorToGraphValueRef,
                nodeIdToGraphValueRef,
                handoffRequirements
        );
    }

    public RegionMemoryPlan {
        structuralView = structuralView == null ? StructuralMemoryView.empty() : structuralView;
        valueLifetimes = Map.copyOf(Objects.requireNonNull(valueLifetimes, "valueLifetimes cannot be null"));
        materializationPlan = Map.copyOf(Objects.requireNonNull(materializationPlan, "materializationPlan cannot be null"));
        memoryBindings = Map.copyOf(Objects.requireNonNull(memoryBindings, "memoryBindings cannot be null"));
        slotByValueRef = Map.copyOf(Objects.requireNonNull(slotByValueRef, "slotByValueRef cannot be null"));
        slotUseCounts = Map.copyOf(Objects.requireNonNull(slotUseCounts, "slotUseCounts cannot be null"));
        slotSizes = Map.copyOf(Objects.requireNonNull(slotSizes, "slotSizes cannot be null"));
        tensorToGraphValueRef = Map.copyOf(Objects.requireNonNull(tensorToGraphValueRef, "tensorToGraphValueRef cannot be null"));
        nodeIdToGraphValueRef = Map.copyOf(Objects.requireNonNull(nodeIdToGraphValueRef, "nodeIdToGraphValueRef cannot be null"));
        handoffRequirements = List.copyOf(Objects.requireNonNull(handoffRequirements, "handoffRequirements cannot be null"));
    }

    public static RegionMemoryPlan empty() {
        return new RegionMemoryPlan(
                StructuralMemoryView.empty(),
                Map.of(),
                Map.of(),
                Map.of(),
                Map.of(),
                Map.of(),
                Map.of(),
                Map.of(),
                List.of()
        );
    }

    private static Map<Integer, Integer> buildSlotUseCounts(Map<GraphValueRef, Integer> slotByValueRef) {
        TreeMap<Integer, Integer> counts = new TreeMap<>();
        for (Integer slotId : slotByValueRef.values()) {
            if (slotId != null) {
                counts.merge(slotId, 1, Integer::sum);
            }
        }
        return counts;
    }
}
