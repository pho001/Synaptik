package backend.accelerator.exec;

import tensor.Tensor;

import java.util.List;

/**
 * External inputs resolved for one accelerator executable invocation.
 *
 * <p>The semantic input ids preserve the native executable ABI. Execution inputs may be the semantic
 * runtime tensors or execution-local prepared tensors created by the CPU layout plan.</p>
 */
public record ResolvedAcceleratorInputs(
        List<Integer> externalInputNodeIds,
        List<Tensor> originalExternalInputs,
        List<Tensor> executionExternalInputs,
        List<Boolean> preparedInputUsed,
        List<AcceleratorPreparedInputSite> resolutionSites
) {
    public ResolvedAcceleratorInputs {
        externalInputNodeIds = List.copyOf(externalInputNodeIds == null ? List.of() : externalInputNodeIds);
        originalExternalInputs = List.copyOf(originalExternalInputs == null ? List.of() : originalExternalInputs);
        executionExternalInputs = List.copyOf(executionExternalInputs == null ? List.of() : executionExternalInputs);
        preparedInputUsed = List.copyOf(preparedInputUsed == null ? List.of() : preparedInputUsed);
        resolutionSites = List.copyOf(resolutionSites == null ? List.of() : resolutionSites);
    }

    public boolean anyPreparedInputUsed() {
        return preparedInputUsed.stream().anyMatch(Boolean::booleanValue);
    }
}
