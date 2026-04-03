package backend.kernels.cpu;

import operations.Operation;
import tensor.Tensor;

import java.util.List;

public class CpuNoopKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (inputs == null || inputs.isEmpty()) return;
        node.aliasRuntimeFrom(inputs.get(0));
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (inputs == null || inputs.isEmpty()) return;
        node.aliasRuntimeFrom(inputs.get(0));
    }

    @Override
    public void forwardF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (inputs == null || inputs.isEmpty()) return;
        node.aliasRuntimeFrom(inputs.get(0));
    }

    @Override
    public void forwardBOOL(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        if (inputs == null || inputs.isEmpty()) return;
        node.aliasRuntimeFrom(inputs.get(0));
    }
}
