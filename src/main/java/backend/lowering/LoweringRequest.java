package backend.lowering;

import planning.memory.MemoryPlan;
import planning.region.PlannedRegion;
import planning.descriptor.CompiledTensorDescriptorIndex;

import java.util.Objects;

/**
 * Inputs required to lower one optimized region into backend executable artifacts.
 *
 * @param region optimized region to lower; must not be {@code null}
 * @param memoryPlan finalized memory plan; must not be {@code null}
 * @param capabilities available backend capabilities; {@code null} becomes no capabilities
 * @param context runtime/partition context; {@code null} becomes an empty context
 */
public record LoweringRequest(
        PlannedRegion region,
        MemoryPlan memoryPlan,
        BackendCapabilities capabilities,
        LoweringContext context
) {
    public LoweringRequest {
        region = Objects.requireNonNull(region, "region cannot be null");
        memoryPlan = Objects.requireNonNull(memoryPlan, "memoryPlan cannot be null");
        capabilities = capabilities == null ? BackendCapabilities.none() : capabilities;
        context = context == null
                ? new LoweringContext(null, java.util.List.of(), CompiledTensorDescriptorIndex.empty(), java.util.Map.of())
                : context;
    }
}
