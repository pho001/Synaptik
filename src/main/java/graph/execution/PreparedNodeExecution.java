package graph.execution;

import graph.CompiledNode;
import tensor.Tensor;

import java.util.Objects;

public record PreparedNodeExecution(
        CompiledNode compiledNode,
        CompiledNodeExecutionMetadata metadata
) {
    public PreparedNodeExecution {
        Objects.requireNonNull(compiledNode, "compiledNode cannot be null");
        Objects.requireNonNull(metadata, "metadata cannot be null");
    }

    public Tensor node() {
        return compiledNode.semanticTensor();
    }
}
