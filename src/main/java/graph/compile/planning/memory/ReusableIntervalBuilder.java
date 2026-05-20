package graph.compile.planning.memory;

import tensor.Tensor;

import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

final class ReusableIntervalBuilder {
    private ReusableIntervalBuilder() {
    }

    static Map<Tensor, ReusableInterval> build(
            List<Tensor> sortedGraph,
            Map<Tensor, NodeLifetime> lifetimes,
            MemoryPlannerPolicy policy
    ) {
        Map<Tensor, ReusableInterval> out = new IdentityHashMap<>();
        for (Tensor tensor : sortedGraph) {
            NodeLifetime lifetime = lifetimes.get(tensor);
            if (lifetime.storageOwner() != tensor) {
                continue;
            }
            if (lifetime.role() != MemoryRole.FORWARD_TEMP
                    && lifetime.role() != MemoryRole.BACKWARD_TEMP
                    && lifetime.role() != MemoryRole.SAVED_FORWARD) {
                continue;
            }
            int size = tensor.getFlatDataSize();
            if (size < policy.minReusableBufferSize()) {
                continue;
            }
            out.put(tensor, new ReusableInterval(
                    tensor,
                    lifetime.birthIndex(),
                    lifetime.lastReadIndex(),
                    size,
                    tensor.getDataType(),
                    lifetime.role()
            ));
        }
        return Map.copyOf(out);
    }
}
