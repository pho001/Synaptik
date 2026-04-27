package graph.execution;

import graph.CompiledNode;
import operations.Operation;
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

    public Operation executionOperation() {
        return metadata.executionOperation() != null ? metadata.executionOperation() : compiledNode.operation();
    }
}
