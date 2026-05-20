package graph.compile.planning.memory;

import backend.runtime.ExecutionMode;
import graph.CompiledNode;
import graph.compile.planning.partition.PartitionPlan;
import graph.compile.planning.region.OptimizedRegion;
import tensor.Tensor;

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

    public static MemoryPlanningInput ofGraph(List<Tensor> graph) {
        List<CompiledNode> nodes = CompiledNode.snapshot(graph == null ? List.of() : graph);
        return new MemoryPlanningInput(
                nodes,
                List.of(),
                Map.of(),
                ExecutionMode.FORWARD,
                false,
                nodes.isEmpty() ? -1 : nodes.size() - 1
        );
    }

    public List<Tensor> graph() {
        return compiledNodes.stream()
                .map(CompiledNode::semanticTensor)
                .toList();
    }

    Tensor forwardOutput() {
        if (compiledNodes.isEmpty()) {
            return null;
        }
        int index = forwardBoundaryNodeId < 0 || forwardBoundaryNodeId >= compiledNodes.size()
                ? compiledNodes.size() - 1
                : forwardBoundaryNodeId;
        return compiledNodes.get(index).semanticTensor();
    }
}
