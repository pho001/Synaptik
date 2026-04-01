package graph.execution;

import tensor.Tensor;

import java.util.Objects;

public record PreparedNodeExecution(
        Tensor node,
        CompiledNodeExecutionMetadata metadata
) {
    public PreparedNodeExecution {
        Objects.requireNonNull(node, "node cannot be null");
        Objects.requireNonNull(metadata, "metadata cannot be null");
    }
}
