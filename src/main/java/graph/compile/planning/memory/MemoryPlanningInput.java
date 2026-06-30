package graph.compile.planning.memory;

import backend.runtime.ExecutionMode;
import graph.model.CompiledNode;
import graph.compile.planning.partition.PartitionPlan;
import graph.compile.planning.region.OptimizedRegion;
import java.util.List;
import java.util.Map;

/**
 * Compile-planning input for memory planning.
 */
public record MemoryPlanningInput(
        List<CompiledNode> compiledNodes,
        List<OptimizedRegion> optimizedRegions,
        Map<String, PartitionPlan> partitionPlansById,
        ExecutionMode executionMode,
        boolean supportsBackward,
        int forwardBoundaryNodeId
) {
    public MemoryPlanningInput {
        compiledNodes = List.copyOf(compiledNodes == null ? List.of() : compiledNodes);
        optimizedRegions = List.copyOf(optimizedRegions == null ? List.of() : optimizedRegions);
        partitionPlansById = Map.copyOf(partitionPlansById == null ? Map.of() : partitionPlansById);
        executionMode = executionMode == null ? ExecutionMode.FORWARD : executionMode;
        if (forwardBoundaryNodeId < -1) {
            throw new IllegalArgumentException("forwardBoundaryNodeId must be >= -1");
        }
    }

}
