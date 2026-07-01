package backend.lowering;

import planning.memory.MemoryPlan;
import planning.partition.ExecutablePartitionPlan;
import planning.descriptor.CompiledTensorDescriptorIndex;

import java.util.Objects;

/**
 * Inputs required to lower one optimized partition into backend executable artifacts.
 *
 * @param executablePartition executable partition to lower; must not be {@code null}
 * @param memoryPlan finalized memory plan; must not be {@code null}
 * @param capabilities available backend capabilities; {@code null} becomes no capabilities
 * @param context runtime/partition context; {@code null} becomes an empty context
 */
public record LoweringRequest(
        ExecutablePartitionPlan executablePartition,
        MemoryPlan memoryPlan,
        BackendCapabilities capabilities,
        LoweringContext context
) {
    public LoweringRequest {
        executablePartition = Objects.requireNonNull(executablePartition, "executablePartition cannot be null");
        memoryPlan = Objects.requireNonNull(memoryPlan, "memoryPlan cannot be null");
        capabilities = capabilities == null ? BackendCapabilities.none() : capabilities;
        context = context == null
                ? new LoweringContext(null, java.util.List.of(), CompiledTensorDescriptorIndex.empty())
                : context;
    }
}
