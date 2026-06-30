package graph.execution.plan;

import backend.contract.ComputeBackend;
import operations.Operation;

import java.util.List;
import java.util.Objects;

/**
 * Prepare-time execution metadata for a compiled node.
 *
 * @param backend backend selected for execution
 * @param executionOperation operation to execute instead of the compiled semantic operation, when present
 * @param executionInputNodeIds node ids to use as execution inputs
 * @param artifact backend-owned prepared execution payload, when applicable
 * @param inputResidencyRequirement backend-neutral input materialization requirement
 * @param outputResidencyEffect backend-neutral output residency effect
 */
public record CompiledNodeExecutionMetadata(
        ComputeBackend backend,
        Operation executionOperation,
        List<Integer> executionInputNodeIds,
        PreparedExecutionArtifact artifact,
        InputResidencyRequirement inputResidencyRequirement,
        OutputResidencyEffect outputResidencyEffect
) {
    public CompiledNodeExecutionMetadata(
            ComputeBackend backend,
            Operation executionOperation,
            List<Integer> executionInputNodeIds,
            PreparedExecutionArtifact artifact
    ) {
        this(
                backend,
                executionOperation,
                executionInputNodeIds,
                artifact,
                defaultInputResidencyRequirement(backend),
                defaultOutputResidencyEffect(backend)
        );
    }

    public CompiledNodeExecutionMetadata {
        Objects.requireNonNull(backend, "backend cannot be null");
        executionInputNodeIds = List.copyOf(executionInputNodeIds == null ? List.of() : executionInputNodeIds);
        inputResidencyRequirement = inputResidencyRequirement == null
                ? defaultInputResidencyRequirement(backend)
                : inputResidencyRequirement;
        outputResidencyEffect = outputResidencyEffect == null
                ? defaultOutputResidencyEffect(backend)
                : outputResidencyEffect;
    }

    private static InputResidencyRequirement defaultInputResidencyRequirement(ComputeBackend backend) {
        if (backend != ComputeBackend.CPU) {
            return InputResidencyRequirement.none();
        }
        return InputResidencyRequirement.cpuReadableAll();
    }

    private static OutputResidencyEffect defaultOutputResidencyEffect(ComputeBackend backend) {
        return backend == ComputeBackend.CPU
                ? OutputResidencyEffect.cpuCurrentPreserveNative()
                : OutputResidencyEffect.cpuCurrentIfUnset("backend wrote CPU array");
    }
}
