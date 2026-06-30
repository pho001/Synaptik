package planning.memory;

import graph.model.CompiledNode;
import planning.partition.PartitionPlan;
import planning.region.PlannedRegion;
import java.util.List;
import java.util.Map;

/**
 * Compile-planning input for memory planning.
 */
public record MemoryPlanningInput(
        List<CompiledNode> compiledNodes,
        List<PlannedRegion> plannedRegions,
        Map<String, PartitionPlan> partitionPlansById,
        boolean supportsBackward,
        int forwardBoundaryNodeId
) {
    public MemoryPlanningInput {
        compiledNodes = List.copyOf(compiledNodes == null ? List.of() : compiledNodes);
        plannedRegions = List.copyOf(plannedRegions == null ? List.of() : plannedRegions);
        partitionPlansById = Map.copyOf(partitionPlansById == null ? Map.of() : partitionPlansById);
        if (forwardBoundaryNodeId < -1) {
            throw new IllegalArgumentException("forwardBoundaryNodeId must be >= -1");
        }
    }

}
