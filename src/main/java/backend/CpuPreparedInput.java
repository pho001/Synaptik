package backend;

import tensor.Tensor;
import tensor.TensorRemap;

import java.util.Objects;

public record CpuPreparedInput(
        int inputIndex,
        Tensor runtimeTensor,
        TensorRemap.RemapPlan remapPlan
) {
    public CpuPreparedInput {
        if (inputIndex < 0) {
            throw new IllegalArgumentException("inputIndex must be >= 0");
        }
        Objects.requireNonNull(runtimeTensor, "runtimeTensor cannot be null");
        Objects.requireNonNull(remapPlan, "remapPlan cannot be null");
    }

    public boolean requiresRemap() {
        return true;
    }
}
