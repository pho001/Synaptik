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
 * @param inputResidencyRequirement backend-neutral input materialization requirement
 * @param outputResidencyEffect backend-neutral output residency effect
 */
public record CompiledNodeExecutionMetadata(
        ComputeBackend backend,
        PartitionExecutionRole partitionRole,
        Operation executionOperation,
        List<Integer> executionInputNodeIds,
        PreparedExecutionArtifact artifact,
        InputResidencyRequirement inputResidencyRequirement,
        OutputResidencyEffect outputResidencyEffect
) {
    public CompiledNodeExecutionMetadata(
            ComputeBackend backend,
            PartitionExecutionRole partitionRole,
            Operation executionOperation,
            List<Integer> executionInputNodeIds,
            PreparedExecutionArtifact artifact
    ) {
        this(
                backend,
                partitionRole,
                executionOperation,
                executionInputNodeIds,
                artifact,
                defaultInputResidencyRequirement(backend, partitionRole),
                defaultOutputResidencyEffect(backend, partitionRole)
        );
    }

    public CompiledNodeExecutionMetadata {
        Objects.requireNonNull(backend, "backend cannot be null");
        partitionRole = partitionRole == null ? PartitionExecutionRole.NONE : partitionRole;
        executionInputNodeIds = List.copyOf(executionInputNodeIds == null ? List.of() : executionInputNodeIds);
        inputResidencyRequirement = inputResidencyRequirement == null
                ? defaultInputResidencyRequirement(backend, partitionRole)
                : inputResidencyRequirement;
        outputResidencyEffect = outputResidencyEffect == null
                ? defaultOutputResidencyEffect(backend, partitionRole)
                : outputResidencyEffect;
    }

    private static InputResidencyRequirement defaultInputResidencyRequirement(
            ComputeBackend backend,
            PartitionExecutionRole role
    ) {
        if (role == PartitionExecutionRole.INTERIOR || backend != ComputeBackend.CPU) {
            return InputResidencyRequirement.none();
        }
        return InputResidencyRequirement.cpuReadableAll();
    }

    private static OutputResidencyEffect defaultOutputResidencyEffect(
            ComputeBackend backend,
            PartitionExecutionRole role
    ) {
        if (role == PartitionExecutionRole.INTERIOR) {
            return OutputResidencyEffect.none();
        }
        return backend == ComputeBackend.CPU
                ? OutputResidencyEffect.cpuCurrentPreserveNative()
                : OutputResidencyEffect.cpuCurrentIfUnset("backend wrote CPU array");
    }
}
