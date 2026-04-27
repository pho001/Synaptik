package backend.cpu.kernels.layout;

import backend.cpu.kernels.*;

import operations.Operation;
import tensor.Tensor;
import java.util.List;

public class CpuContiguousKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        LayoutExecutor.contiguous(inputs, node, context);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        LayoutExecutor.contiguous(inputs, node, context);
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        LayoutExecutor.contiguous(inputs, node, context);
    }

    @Override
    public void forwardBOOL(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        LayoutExecutor.contiguous(inputs, node, context);
    }

    @Override
    public void forwardI32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        LayoutExecutor.contiguous(inputs, node, context);
    }
}
