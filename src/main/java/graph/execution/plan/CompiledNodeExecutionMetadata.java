package graph.execution.plan;

import backend.ComputeBackend;
import backend.accelerator.exec.PartitionExecutionRole;
import operations.Operation;

import java.util.List;
import java.util.Objects;

/**
 * Prepare-time execution metadata for a compiled node.
 *
 * @param backend backend selected for execution
 * @param partitionRole role of this node in partitioned execution
 * @param executionOperation operation to execute instead of the compiled semantic operation, when present
 * @param executionInputNodeIds node ids to use as execution inputs
 * @param artifact backend-owned prepared execution payload, when applicable
 */
public record CompiledNodeExecutionMetadata(
        ComputeBackend backend,
        PartitionExecutionRole partitionRole,
        Operation executionOperation,
        List<Integer> executionInputNodeIds,
        PreparedExecutionArtifact artifact
) {
    public CompiledNodeExecutionMetadata {
        Objects.requireNonNull(backend, "backend cannot be null");
        partitionRole = partitionRole == null ? PartitionExecutionRole.NONE : partitionRole;
        executionInputNodeIds = List.copyOf(executionInputNodeIds == null ? List.of() : executionInputNodeIds);
    }
}
