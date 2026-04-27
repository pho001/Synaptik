package graph.optimizer.state;

import backend.runtime.ExecutionMode;
import graph.optimizer.memory.MemoryPlan;
import graph.optimizer.partition.Partition;
import graph.optimizer.partition.PartitionPlan;
import graph.optimizer.region.OptimizedRegion;
import tensor.Tensor;

import java.util.List;
import java.util.Map;
import java.util.Objects;

public record OptimizerState(
        List<Tensor> graph,
        Tensor forwardOutput,
        ExecutionMode executionMode,
        boolean supportsBackward,
        int forwardBoundaryNodeId,
        List<Partition> partitions,
        Map<String, PartitionPlan> partitionPlansById,
        List<OptimizedRegion> optimizedRegions,
        MemoryPlan memoryPlan,
        OptimizerTrace trace
) {
    public OptimizerState {
        graph = List.copyOf(Objects.requireNonNull(graph, "graph cannot be null"));
        forwardOutput = Objects.requireNonNull(forwardOutput, "forwardOutput cannot be null");
        executionMode = executionMode == null ? ExecutionMode.FORWARD : executionMode;
        partitions = List.copyOf(partitions == null ? List.of() : partitions);
        partitionPlansById = Map.copyOf(partitionPlansById == null ? Map.of() : partitionPlansById);
        optimizedRegions = List.copyOf(optimizedRegions == null ? List.of() : optimizedRegions);
        trace = trace == null ? OptimizerTrace.empty() : trace;
    }

    public static OptimizerState ofGraph(List<Tensor> graph) {
        List<Tensor> safe = List.copyOf(Objects.requireNonNull(graph, "graph cannot be null"));
        if (safe.isEmpty()) {
            throw new IllegalArgumentException("graph cannot be empty");
        }
        Tensor forwardOutput = safe.getLast();
        return ofGraph(safe, forwardOutput);
    }

    public static OptimizerState ofGraph(List<Tensor> graph, Tensor forwardOutput) {
        List<Tensor> safe = List.copyOf(Objects.requireNonNull(graph, "graph cannot be null"));
        Tensor resolvedForwardOutput = Objects.requireNonNull(forwardOutput, "forwardOutput cannot be null");
        int forwardBoundaryNodeId = safe.indexOf(resolvedForwardOutput);
        if (forwardBoundaryNodeId < 0) {
            throw new IllegalArgumentException("forwardOutput must be part of graph");
        }
        return new OptimizerState(
                safe,
                resolvedForwardOutput,
                ExecutionMode.FORWARD,
                false,
                forwardBoundaryNodeId,
                List.of(),
                Map.of(),
                List.of(),
                null,
                OptimizerTrace.empty()
        );
    }

    public OptimizerState withGraph(List<Tensor> graph, Tensor forwardOutput) {
        return new OptimizerState(
                graph,
                forwardOutput,
                executionMode,
                supportsBackward,
                resolveBoundaryIndex(graph, forwardOutput),
                List.of(),
                Map.of(),
                List.of(),
                null,
                trace
        );
    }

    public OptimizerState withPartitions(List<Partition> partitions) {
        return withPartitions(partitions, Map.of());
    }

    public OptimizerState withPartitions(List<Partition> partitions, Map<String, PartitionPlan> partitionPlansById) {
        return new OptimizerState(
                graph,
                forwardOutput,
                executionMode,
                supportsBackward,
                forwardBoundaryNodeId,
                partitions,
                partitionPlansById,
                List.of(),
                null,
                trace
        );
    }

    public OptimizerState withOptimizedRegions(List<OptimizedRegion> regions) {
        return new OptimizerState(
                graph,
                forwardOutput,
                executionMode,
                supportsBackward,
                forwardBoundaryNodeId,
                partitions,
                partitionPlansById,
                regions,
                null,
                trace
        );
    }

    public OptimizerState withMemoryPlan(MemoryPlan memoryPlan) {
        return new OptimizerState(
                graph,
                forwardOutput,
                executionMode,
                supportsBackward,
                forwardBoundaryNodeId,
                partitions,
                partitionPlansById,
                optimizedRegions,
                memoryPlan,
                trace
        );
    }

    public OptimizerState withTrace(OptimizerTrace trace) {
        return new OptimizerState(
                graph,
                forwardOutput,
                executionMode,
                supportsBackward,
                forwardBoundaryNodeId,
                partitions,
                partitionPlansById,
                optimizedRegions,
                memoryPlan,
                trace
        );
    }

    public OptimizerState withExecutionMetadata(
            ExecutionMode executionMode,
            boolean supportsBackward,
            int forwardBoundaryNodeId
    ) {
        return new OptimizerState(
                graph,
                forwardOutput,
                executionMode,
                supportsBackward,
                forwardBoundaryNodeId,
                partitions,
                partitionPlansById,
                optimizedRegions,
                memoryPlan,
                trace
        );
    }

    private static int resolveBoundaryIndex(List<Tensor> graph, Tensor forwardOutput) {
        List<Tensor> safe = List.copyOf(Objects.requireNonNull(graph, "graph cannot be null"));
        Tensor resolvedForwardOutput = Objects.requireNonNull(forwardOutput, "forwardOutput cannot be null");
        int index = safe.indexOf(resolvedForwardOutput);
        if (index < 0) {
            throw new IllegalArgumentException("forwardOutput must be part of graph");
        }
        return index;
    }
}
