package backend.cpu.kernels.layout;

import backend.cpu.plan.CpuPreparedInput;
import tensor.Tensor;

import java.util.List;

public record PreparedInputsResult(
        List<CpuPreparedInput> preparedInputs,
        List<Tensor> runtimeInputs
) {
    public PreparedInputsResult {
        preparedInputs = List.copyOf(preparedInputs == null ? List.of() : preparedInputs);
        runtimeInputs = List.copyOf(runtimeInputs == null ? List.of() : runtimeInputs);
    }
}
