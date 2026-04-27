package backend.lowering;

import graph.optimizer.memory.MemoryPlan;
import graph.optimizer.region.OptimizedRegion;

import java.util.Objects;

public record LoweringRequest(
        OptimizedRegion region,
        MemoryPlan memoryPlan,
        BackendCapabilities capabilities,
        LoweringContext context
) {
    public LoweringRequest {
        region = Objects.requireNonNull(region, "region cannot be null");
        memoryPlan = Objects.requireNonNull(memoryPlan, "memoryPlan cannot be null");
        capabilities = capabilities == null ? BackendCapabilities.none() : capabilities;
        context = context == null ? new LoweringContext(null, java.util.List.of()) : context;
    }
}
