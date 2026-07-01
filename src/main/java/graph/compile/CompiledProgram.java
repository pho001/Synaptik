package graph.compile;

import planning.descriptor.CompiledTensorDescriptorIndex;
import planning.memory.MemoryPlan;
import planning.partition.Partition;
import planning.partition.PartitionPlan;
import planning.partition.ExecutablePartitionPlan;
import planning.partition.PlannedPartition;
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
        boolean partitionExecutionEnabled,
        List<ExecutablePartitionPlan> executablePartitions,
        MemoryPlan memoryPlan
) {
    public CompiledProgram {
        compiledNodes = List.copyOf(compiledNodes == null ? List.of() : compiledNodes);
        descriptorIndex = Objects.requireNonNull(descriptorIndex, "descriptorIndex cannot be null");
        executablePartitions = List.copyOf(executablePartitions == null ? List.of() : executablePartitions);
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
        return executablePartitions.stream()
                .map(ExecutablePartitionPlan::partition)
                .toList();
    }

    public List<PlannedPartition> plannedPartitions() {
        return executablePartitions.stream()
                .map(ExecutablePartitionPlan::plannedPartition)
                .toList();
    }

    public List<PartitionPlan> backendPlans() {
        return executablePartitions.stream()
                .map(ExecutablePartitionPlan::backendPlan)
                .filter(Objects::nonNull)
                .toList();
    }
}
