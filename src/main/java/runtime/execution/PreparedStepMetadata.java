package runtime.execution;

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
 * @param executable backend-owned prepared executable
 * @param inputResidencyRequirement backend-neutral input materialization requirement
 * @param outputResidencyEffect backend-neutral output residency effect
 */
public record PreparedStepMetadata(
        ComputeBackend backend,
        Operation executionOperation,
        List<Integer> executionInputNodeIds,
        PreparedStepExecutable executable,
        InputResidencyRequirement inputResidencyRequirement,
        OutputResidencyEffect outputResidencyEffect
) {
    public PreparedStepMetadata {
        Objects.requireNonNull(backend, "backend cannot be null");
        Objects.requireNonNull(executable, "executable cannot be null");
        executionInputNodeIds = List.copyOf(executionInputNodeIds == null ? List.of() : executionInputNodeIds);
        Objects.requireNonNull(inputResidencyRequirement, "inputResidencyRequirement cannot be null");
        Objects.requireNonNull(outputResidencyEffect, "outputResidencyEffect cannot be null");
    }
}
