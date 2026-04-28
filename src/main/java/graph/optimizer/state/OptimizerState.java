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

/**
 * Immutable state passed between optimizer stages.
 *
 * <p>The graph and forward output are the primary contract. Later stages attach increasingly specific planning data:
 * partition plans, optimized regions, memory plans, and trace events. The {@code with...} methods intentionally clear
 * downstream products when an upstream product changes so stale partition, fusion, or memory metadata is not reused.
 *
 * @param graph tensors in topological order
 * @param forwardOutput semantic forward output that must remain observable
 * @param executionMode compile-time execution mode metadata
 * @param supportsBackward whether backward work is present in the graph
 * @param forwardBoundaryNodeId index of the forward output node in {@code graph}
 * @param partitions accepted backend partitions
 * @param partitionPlansById backend plans keyed by partition id
 * @param optimizedRegions region/fusion plans derived from partitions
 * @param memoryPlan memory reuse and region handoff plan
 * @param trace optimizer diagnostics
 */
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

    /**
     * Creates optimizer state using the last tensor as the forward output.
     *
     * @param graph tensors in topological order
     * @return optimizer state with empty downstream planning data
     */
    public static OptimizerState ofGraph(List<Tensor> graph) {
        List<Tensor> safe = List.copyOf(Objects.requireNonNull(graph, "graph cannot be null"));
        if (safe.isEmpty()) {
            throw new IllegalArgumentException("graph cannot be empty");
        }
        Tensor forwardOutput = safe.getLast();
        return ofGraph(safe, forwardOutput);
    }

    /**
     * Creates optimizer state with an explicit forward output.
     *
     * @param graph tensors in topological order
     * @param forwardOutput semantic forward output contained in {@code graph}
     * @return optimizer state with empty downstream planning data
     */
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

    /**
     * Returns a copy with a replacement graph and cleared partition, region, and memory products.
     *
     * @param graph replacement graph in topological order
     * @param forwardOutput replacement semantic forward output contained in {@code graph}
     * @return updated optimizer state
     */
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

    /**
     * Returns a copy with partitions and no backend-specific plans.
     *
     * @param partitions accepted partitions
     * @return updated optimizer state with optimized regions and memory plan cleared
     */
    public OptimizerState withPartitions(List<Partition> partitions) {
        return withPartitions(partitions, Map.of());
    }

    /**
     * Returns a copy with partitions and backend plans.
     *
     * @param partitions accepted partitions
     * @param partitionPlansById backend plans keyed by partition id
     * @return updated optimizer state with optimized regions and memory plan cleared
     */
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

    /**
     * Returns a copy with optimized regions and memory plan cleared.
     *
     * @param regions optimized region plans
     * @return updated optimizer state
     */
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

    /**
     * Returns a copy with an attached memory plan.
     *
     * @param memoryPlan memory plan, or {@code null} when memory planning is disabled
     * @return updated optimizer state
     */
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

    /**
     * Returns a copy with replacement optimizer trace metadata.
     *
     * @param trace trace metadata
     * @return updated optimizer state
     */
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

    /**
     * Returns a copy with compile-time execution metadata.
     *
     * @param executionMode execution mode represented by the compiled graph
     * @param supportsBackward whether backward work is present
     * @param forwardBoundaryNodeId index of the last forward node
     * @return updated optimizer state
     */
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
