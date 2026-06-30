package planning.memory;

import tensor.Tensor;

import java.util.Map;
import java.util.Objects;

public record RuntimeBindingPlan(
        Map<Tensor, RuntimeMemoryBindingPolicy> policiesByTensor,
        Map<Integer, RuntimeMemoryBindingPolicy> policiesByNodeId
) {
    public RuntimeBindingPlan {
        policiesByTensor = Map.copyOf(Objects.requireNonNull(policiesByTensor, "policiesByTensor cannot be null"));
        policiesByNodeId = Map.copyOf(Objects.requireNonNull(policiesByNodeId, "policiesByNodeId cannot be null"));
    }

    public static RuntimeBindingPlan empty() {
        return new RuntimeBindingPlan(Map.of(), Map.of());
    }
}
