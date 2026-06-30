package graph.compile;

import graph.compile.descriptor.CompiledTensorDescriptorIndex;
import graph.compile.planning.memory.MemoryPlan;
import graph.compile.planning.partition.Partition;
import graph.compile.planning.partition.PartitionPlan;
import graph.compile.planning.partition.PlannedPartition;
import graph.compile.planning.region.OptimizedRegion;
import graph.model.CompiledNode;

import java.util.List;
import java.util.Objects;

/**
 * Executable compile-time program snapshot.
 *
 * <p>This model owns runtime executable graph facts only. Publication tensors, graph contracts, and diagnostics live
 * outside the program.</p>
 */
public record CompiledProgram(
        List<CompiledNode> compiledNodes,
        CompiledTensorDescriptorIndex descriptorIndex,
        int forwardOutputNodeId,
        int forwardBoundaryNodeId,
        boolean supportsBackward,
        List<PlannedPartition> plannedPartitions,
        List<OptimizedRegion> optimizedRegions,
        MemoryPlan memoryPlan
) {
    public CompiledProgram {
        compiledNodes = List.copyOf(compiledNodes == null ? List.of() : compiledNodes);
        descriptorIndex = Objects.requireNonNull(descriptorIndex, "descriptorIndex cannot be null");
        plannedPartitions = List.copyOf(plannedPartitions == null ? List.of() : plannedPartitions);
        optimizedRegions = List.copyOf(optimizedRegions == null ? List.of() : optimizedRegions);
        if (forwardOutputNodeId < 0 && !compiledNodes.isEmpty()) {
            throw new IllegalArgumentException("forwardOutputNodeId must be >= 0");
        }
        if (forwardBoundaryNodeId < 0 && !compiledNodes.isEmpty()) {
            throw new IllegalArgumentException("forwardBoundaryNodeId must be >= 0");
        }
    }

    public CompiledNode forwardOutputNode() {
        if (forwardOutputNodeId < 0 || forwardOutputNodeId >= compiledNodes.size()) {
            throw new IllegalStateException("Forward output node id is outside the compiled program.");
        }
        return compiledNodes.get(forwardOutputNodeId);
    }

    public List<Partition> partitions() {
        return plannedPartitions.stream()
                .map(PlannedPartition::partition)
                .toList();
    }

    public List<PartitionPlan> backendPlans() {
        return plannedPartitions.stream()
                .map(PlannedPartition::plan)
                .filter(Objects::nonNull)
                .toList();
    }
}
