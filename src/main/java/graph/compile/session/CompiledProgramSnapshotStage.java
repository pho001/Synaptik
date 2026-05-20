package graph.compile.session;

import graph.CompiledNode;
import graph.compile.descriptor.CompiledTensorDescriptorBuilder;
import graph.compile.descriptor.CompiledTensorDescriptorIndex;
import graph.compile.intent.BackendIntentPropagator;
import tensor.Tensor;

import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Freezes optimized tensors into compiled-node and descriptor snapshots.
 */
final class CompiledProgramSnapshotStage {
    private CompiledProgramSnapshotStage() {
    }

    record Result(
            List<Tensor> graph,
            List<CompiledNode> compiledNodes,
            CompiledTensorDescriptorIndex descriptorIndex,
            Map<Tensor, CompiledNode> compiledNodeByTensor,
            CompiledNode forwardOutput,
            int forwardBoundaryNodeId
    ) {
        public Result {
            graph = List.copyOf(graph == null ? List.of() : graph);
            compiledNodes = List.copyOf(compiledNodes == null ? List.of() : compiledNodes);
            descriptorIndex = Objects.requireNonNull(descriptorIndex, "descriptorIndex cannot be null");
            compiledNodeByTensor = identityCopy(compiledNodeByTensor);
            forwardOutput = Objects.requireNonNull(forwardOutput, "forwardOutput cannot be null");
            if (forwardBoundaryNodeId < 0) {
                throw new IllegalArgumentException("forwardBoundaryNodeId must be >= 0");
            }
        }
    }

    static Result snapshot(List<Tensor> graph, Tensor forwardOutput, String graphDescription) {
        List<Tensor> finalGraph = List.copyOf(graph == null ? List.of() : graph);
        int forwardBoundaryNodeId = finalGraph.indexOf(forwardOutput);
        if (forwardBoundaryNodeId == -1) {
            String description = graphDescription == null ? "finalGraph" : graphDescription;
            throw new IllegalStateException("Forward output node not found in " + description + ".");
        }

        BackendIntentPropagator.propagateBackwardClosure(finalGraph);
        List<CompiledNode> compiledNodes = CompiledNode.snapshot(finalGraph);
        CompiledTensorDescriptorIndex descriptorIndex = CompiledTensorDescriptorBuilder.build(compiledNodes);

        IdentityHashMap<Tensor, CompiledNode> index = new IdentityHashMap<>();
        for (int i = 0; i < compiledNodes.size(); i++) {
            index.put(finalGraph.get(i), compiledNodes.get(i));
        }
        CompiledNode compiledForwardOutput = index.get(forwardOutput);
        if (compiledForwardOutput == null) {
            throw new IllegalStateException("Forward output compiled node snapshot is missing.");
        }

        return new Result(
                finalGraph,
                compiledNodes,
                descriptorIndex,
                index,
                compiledForwardOutput,
                forwardBoundaryNodeId
        );
    }

    private static Map<Tensor, CompiledNode> identityCopy(Map<Tensor, CompiledNode> source) {
        IdentityHashMap<Tensor, CompiledNode> copy = new IdentityHashMap<>();
        if (source != null) {
            copy.putAll(source);
        }
        return Collections.unmodifiableMap(copy);
    }
}
