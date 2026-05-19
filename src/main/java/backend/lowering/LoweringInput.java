package backend.lowering;

import graph.optimizer.memory.MemoryPlan;
import graph.optimizer.partition.PartitionPlan;
import graph.optimizer.region.OptimizedRegion;

import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Minimal compile-time input required by backend region lowering.
 *
 * <p>This intentionally avoids exposing optimizer pipeline state to prepare-time lowering. The optimizer may still
 * produce these artifacts, but lowering consumes only finalized regions, memory plan, and selected partition plans.
 *
 * @param optimizedRegions optimized regions to lower
 * @param memoryPlan finalized memory plan
 * @param partitionPlansById backend plans keyed by partition id
 */
public record LoweringInput(
        List<OptimizedRegion> optimizedRegions,
        MemoryPlan memoryPlan,
        Map<String, PartitionPlan> partitionPlansById
) {
    public LoweringInput {
        optimizedRegions = List.copyOf(optimizedRegions == null ? List.of() : optimizedRegions);
        memoryPlan = Objects.requireNonNull(memoryPlan, "memoryPlan cannot be null");
        partitionPlansById = Map.copyOf(partitionPlansById == null ? Map.of() : partitionPlansById);
    }

    public LoweringInput withPartitionPlans(Map<String, PartitionPlan> partitionPlansById) {
        return new LoweringInput(optimizedRegions, memoryPlan, partitionPlansById);
    }
}
