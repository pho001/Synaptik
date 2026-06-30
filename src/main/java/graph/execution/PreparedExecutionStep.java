package graph.execution;

import graph.model.CompiledNode;
import graph.execution.plan.CompiledNodeExecutionMetadata;
import operations.Operation;

import java.util.List;
import java.util.Objects;

/**
 * Prepared execution step for one prepared runtime unit.
 *
 * @param compiledNode compile-time output node represented by this step
 * @param metadata runtime execution metadata selected during preparation
 * @param orderedNodeIds compiled node ids covered by this step in execution order
 * @param boundaryOutputNodeIds output node ids published by this step
 */
public record PreparedExecutionStep(
        CompiledNode compiledNode,
        CompiledNodeExecutionMetadata metadata,
        List<Integer> orderedNodeIds,
        List<Integer> boundaryOutputNodeIds
) {
    public PreparedExecutionStep {
        Objects.requireNonNull(compiledNode, "compiledNode cannot be null");
        Objects.requireNonNull(metadata, "metadata cannot be null");
        orderedNodeIds = List.copyOf(orderedNodeIds == null ? List.of(compiledNode.id()) : orderedNodeIds);
        boundaryOutputNodeIds = List.copyOf(boundaryOutputNodeIds == null ? List.of(compiledNode.id()) : boundaryOutputNodeIds);
        if (orderedNodeIds.isEmpty()) {
            throw new IllegalArgumentException("orderedNodeIds cannot be empty");
        }
        if (boundaryOutputNodeIds.isEmpty()) {
            throw new IllegalArgumentException("boundaryOutputNodeIds cannot be empty");
        }
        if (!boundaryOutputNodeIds.contains(compiledNode.id())) {
            throw new IllegalArgumentException("boundaryOutputNodeIds must contain compiledNode.id=" + compiledNode.id());
        }
    }

    public PreparedExecutionStep(CompiledNode compiledNode, CompiledNodeExecutionMetadata metadata) {
        this(compiledNode, metadata, List.of(compiledNode.id()), List.of(compiledNode.id()));
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
