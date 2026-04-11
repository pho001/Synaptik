package backend.kernels.cpu;

import backend.kernels.cpu.elementwise.WhereExecutor;
import operations.Operation;
import tensor.Tensor;

import java.util.List;

public final class CpuWhereKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        WhereExecutor.execute(inputs, node, context);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        WhereExecutor.execute(inputs, node, context);
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        WhereExecutor.execute(inputs, node, context);
    }
}
