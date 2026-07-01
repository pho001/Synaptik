package backend.lowering;

import planning.memory.MemoryPlan;
import planning.partition.ExecutablePartitionPlan;

import java.util.List;
import java.util.Objects;

/**
 * Minimal compile-time input required by backend partition lowering.
 *
 * <p>This intentionally avoids exposing optimizer pipeline state to prepare-time lowering. The optimizer may still
 * produce these artifacts, but lowering consumes only finalized partitions, memory plan, and selected partition plans.
 *
 * @param executablePartitions executable partitions to lower
 * @param memoryPlan finalized memory plan
 */
public record LoweringInput(
        List<ExecutablePartitionPlan> executablePartitions,
        MemoryPlan memoryPlan
) {
    public LoweringInput {
        executablePartitions = List.copyOf(executablePartitions == null ? List.of() : executablePartitions);
        memoryPlan = Objects.requireNonNull(memoryPlan, "memoryPlan cannot be null");
    }
}
