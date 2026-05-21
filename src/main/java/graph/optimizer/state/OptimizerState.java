package graph.optimizer.state;

import backend.runtime.ExecutionMode;
import tensor.Tensor;

import java.util.List;
import java.util.Objects;

/**
 * Immutable state passed between backend-neutral graph optimizer stages.
 *
 * <p>The graph and forward output are the primary contract. Compile-planning artifacts such as backend partitions,
 * optimized regions, and memory plans live outside optimizer state so graph rewrites cannot carry stale backend
 * planning metadata.
 *
 * @param graph tensors in topological order
 * @param forwardOutput semantic forward output that must remain observable
 * @param executionMode compile-time execution mode metadata
 * @param supportsBackward whether backward work is present in the graph
 * @param forwardBoundaryNodeId index of the forward output node in {@code graph}
 * @param rewriteMap cumulative identity mapping from pre-optimization tensors to rewritten tensors
 * @param trace optimizer diagnostics
 */
public record OptimizerState(
        List<Tensor> graph,
        Tensor forwardOutput,
        ExecutionMode executionMode,
        boolean supportsBackward,
        int forwardBoundaryNodeId,
        GraphRewriteMap rewriteMap,
        OptimizerTrace trace
) {
    public OptimizerState {
        graph = List.copyOf(Objects.requireNonNull(graph, "graph cannot be null"));
        forwardOutput = Objects.requireNonNull(forwardOutput, "forwardOutput cannot be null");
        executionMode = executionMode == null ? ExecutionMode.FORWARD : executionMode;
        rewriteMap = rewriteMap == null ? GraphRewriteMap.empty() : rewriteMap;
        trace = trace == null ? OptimizerTrace.empty() : trace;
    }

    public static OptimizerState ofGraph(List<Tensor> graph) {
        List<Tensor> safe = List.copyOf(Objects.requireNonNull(graph, "graph cannot be null"));
        if (safe.isEmpty()) {
            throw new IllegalArgumentException("graph cannot be empty");
        }
        return ofGraph(safe, safe.getLast());
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
                GraphRewriteMap.empty(),
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
                rewriteMap,
                trace
        );
    }

    public OptimizerState withRewriteMap(GraphRewriteMap rewriteMap) {
        return new OptimizerState(
                graph,
                forwardOutput,
                executionMode,
                supportsBackward,
                forwardBoundaryNodeId,
                rewriteMap,
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
                rewriteMap,
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
                rewriteMap,
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
