package backend.cpu.kernels.reduction;

import graph.compile.descriptor.CompiledTensorDescriptor;

import java.util.List;

public final class ReductionLogicalSize {
    private ReductionLogicalSize() {
    }

    public static int estimate(List<CompiledTensorDescriptor> runtimeInputs, CompiledTensorDescriptor node) {
        if (runtimeInputs != null && !runtimeInputs.isEmpty() && runtimeInputs.get(0) != null) {
            return Math.toIntExact(runtimeInputs.get(0).logicalElementCount());
        }
        return Math.toIntExact(node.logicalElementCount());
    }
}
