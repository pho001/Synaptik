package backend.accelerator.buffer;

import backend.ComputeBackend;
import tensor.DataType;

import java.util.List;
import java.util.Objects;

/**
 * Backend-neutral facts needed to decide whether native buffer bindings are legal.
 */
public record AcceleratorBufferRequest(
        ComputeBackend backend,
        long estimatedWork,
        List<Integer> externalInputNodeIds,
        List<DataType> externalInputDataTypes,
        List<Integer> outputNodeIds,
        List<DataType> outputDataTypes,
        boolean runsBackwardPass
) {
    public AcceleratorBufferRequest {
        Objects.requireNonNull(backend, "backend cannot be null");
        estimatedWork = Math.max(0L, estimatedWork);
        externalInputNodeIds = List.copyOf(externalInputNodeIds == null ? List.of() : externalInputNodeIds);
        externalInputDataTypes = List.copyOf(externalInputDataTypes == null ? List.of() : externalInputDataTypes);
        outputNodeIds = List.copyOf(outputNodeIds == null ? List.of() : outputNodeIds);
        outputDataTypes = List.copyOf(outputDataTypes == null ? List.of() : outputDataTypes);
    }
}
