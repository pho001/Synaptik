package backend.kernels.cpu.elementwise.compare;

import backend.kernels.cpu.CpuKernel;
import backend.kernels.cpu.CpuKernelContext;
import operations.Operation;
import tensor.Tensor;

import java.util.List;

public final class CpuLessThanKernel implements CpuKernel, CompareElementwiseKernel {
    @Override
    public void forwardBOOL(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        CompareExecutor.execute(this, inputs, node, context);
    }

    @Override
    public boolean testF64(double left, double right) {
        return left < right;
    }

    @Override
    public boolean testF32(float left, float right) {
        return left < right;
    }

    @Override
    public boolean testBF16(float left, float right) {
        return left < right;
    }
}
