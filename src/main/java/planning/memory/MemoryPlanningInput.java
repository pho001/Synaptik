package planning.memory;

import graph.model.CompiledNode;
import planning.partition.ExecutablePartitionPlan;
import java.util.List;

/**
 * Compile-planning input for memory planning.
 */
public record MemoryPlanningInput(
        List<CompiledNode> compiledNodes,
        List<ExecutablePartitionPlan> executablePartitions,
        boolean supportsBackward,
        int forwardBoundaryNodeId
) {
    public MemoryPlanningInput {
        compiledNodes = List.copyOf(compiledNodes == null ? List.of() : compiledNodes);
        executablePartitions = List.copyOf(executablePartitions == null ? List.of() : executablePartitions);
        if (forwardBoundaryNodeId < -1) {
            throw new IllegalArgumentException("forwardBoundaryNodeId must be >= -1");
        }
    }

}
