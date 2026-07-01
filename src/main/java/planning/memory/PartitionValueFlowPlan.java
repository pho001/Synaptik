package planning.memory;

import planning.value.GraphValueRef;
import tensor.Tensor;

import java.util.List;
import java.util.Map;

record PartitionValueFlowPlan(
        StructuralMemoryView structuralView,
        Map<GraphValueRef, PartitionValueLifetime> partitionValueLifetimes,
        Map<GraphValueRef, MaterializationPlanEntry> materializationPlan,
        Map<Tensor, GraphValueRef> tensorToGraphValueRef,
        Map<Integer, GraphValueRef> nodeIdToGraphValueRef
) {
    static PartitionValueFlowPlan empty() {
        return new PartitionValueFlowPlan(
                StructuralMemoryView.empty(),
                Map.of(),
                Map.of(),
                Map.of(),
                Map.of()
        );
    }
}
