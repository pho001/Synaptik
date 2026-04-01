package backend.kernels.cpu;

import operations.Operation;
import tensor.Tensor;

import java.util.List;

public class CpuPermuteKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        forwardAny(inputs, node);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        forwardAny(inputs, node);
    }

    @Override
    public void forwardF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        forwardAny(inputs, node);
    }

    private static void forwardAny(List<Tensor> inputs, Tensor node) {
        if (inputs == null || inputs.isEmpty()) {
            return;
        }
        node.aliasRuntimeFrom(inputs.getFirst());
    }
}
