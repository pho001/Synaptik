package graph.execution;

import graph.CompiledNode;
import graph.execution.plan.CompiledNodeExecutionMetadata;
import operations.Operation;

import java.util.Objects;

/**
 * Prepared execution step for one compiled node.
 *
 * @param compiledNode compile-time node snapshot
 * @param metadata runtime execution metadata selected during preparation
 */
public record PreparedNodeExecution(
        CompiledNode compiledNode,
        CompiledNodeExecutionMetadata metadata
) {
    public PreparedNodeExecution {
        Objects.requireNonNull(compiledNode, "compiledNode cannot be null");
        Objects.requireNonNull(metadata, "metadata cannot be null");
    }

    /**
     * Returns the operation that will execute for this step.
     *
     * @return prepared execution operation when present, otherwise the compiled node operation
     */
    public Operation executionOperation() {
        return metadata.executionOperation() != null ? metadata.executionOperation() : compiledNode.operation();
    }
}
