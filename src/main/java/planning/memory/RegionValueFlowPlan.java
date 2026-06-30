package planning.memory;

import planning.value.GraphValueRef;
import tensor.Tensor;

import java.util.List;
import java.util.Map;

record RegionValueFlowPlan(
        StructuralMemoryView structuralView,
        Map<GraphValueRef, RegionValueLifetime> regionValueLifetimes,
        Map<GraphValueRef, MaterializationPlanEntry> materializationPlan,
        Map<Tensor, GraphValueRef> tensorToGraphValueRef,
        Map<Integer, GraphValueRef> nodeIdToGraphValueRef
) {
    static RegionValueFlowPlan empty() {
        return new RegionValueFlowPlan(
                StructuralMemoryView.empty(),
                Map.of(),
                Map.of(),
                Map.of(),
                Map.of()
        );
    }
}
