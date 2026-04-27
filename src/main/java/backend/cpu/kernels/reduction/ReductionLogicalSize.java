package backend.cpu.kernels.reduction;

import tensor.Tensor;

import java.util.List;

public final class ReductionLogicalSize {
    private ReductionLogicalSize() {
    }

    public static int estimate(List<Tensor> runtimeInputs, Tensor node) {
        if (runtimeInputs != null && !runtimeInputs.isEmpty() && runtimeInputs.get(0) != null) {
            return runtimeInputs.get(0).getFlatDataSize();
        }
        return node.getFlatDataSize();
    }
}
