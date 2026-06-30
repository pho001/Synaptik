package backend.lowering;

import planning.memory.MemoryPlan;
import planning.partition.PartitionPlan;
import planning.region.PlannedRegion;

import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Minimal compile-time input required by backend region lowering.
 *
 * <p>This intentionally avoids exposing optimizer pipeline state to prepare-time lowering. The optimizer may still
 * produce these artifacts, but lowering consumes only finalized regions, memory plan, and selected partition plans.
 *
 * @param plannedRegions optimized regions to lower
 * @param memoryPlan finalized memory plan
 * @param partitionPlansById backend plans keyed by partition id
 */
public record LoweringInput(
        List<PlannedRegion> plannedRegions,
        MemoryPlan memoryPlan,
        Map<String, PartitionPlan> partitionPlansById
) {
    public LoweringInput {
        plannedRegions = List.copyOf(plannedRegions == null ? List.of() : plannedRegions);
        memoryPlan = Objects.requireNonNull(memoryPlan, "memoryPlan cannot be null");
        partitionPlansById = Map.copyOf(partitionPlansById == null ? Map.of() : partitionPlansById);
    }

    public LoweringInput withPartitionPlans(Map<String, PartitionPlan> partitionPlansById) {
        return new LoweringInput(plannedRegions, memoryPlan, partitionPlansById);
    }
}
