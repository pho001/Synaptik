package backend.cpu.plan.layout;

import backend.cpu.plan.CpuPreparedInput;
import graph.compile.descriptor.CompiledTensorDescriptor;

import java.util.List;

public record PreparedInputsResult(
        List<CpuPreparedInput> preparedInputs,
        List<CompiledTensorDescriptor> runtimeInputDescriptors
) {
    public PreparedInputsResult {
        preparedInputs = List.copyOf(preparedInputs == null ? List.of() : preparedInputs);
        runtimeInputDescriptors = List.copyOf(runtimeInputDescriptors == null ? List.of() : runtimeInputDescriptors);
    }
}
