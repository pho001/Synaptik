package backend.kernels.cpu.linalg;

import backend.kernels.cpu.CpuKernel;
import backend.kernels.cpu.CpuKernelContext;
import backend.kernels.cpu.CpuKernelCostClass;
import operations.Operation;
import tensor.Tensor;

import java.util.List;

public class CpuMatMulKernel implements CpuKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        MatMulExecutor.forwardF64(inputs.get(0), inputs.get(1), node, context);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        MatMulExecutor.forwardF32(inputs.get(0), inputs.get(1), node, context);
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        MatMulExecutor.forwardBF16(inputs.get(0), inputs.get(1), node, context);
    }

    @Override
    public CpuKernelCostClass costClass(Operation op) {
        return CpuKernelCostClass.HIGH;
    }
}
