package backend.cpu.kernels.elementwise.compare;

import backend.cpu.kernels.TypedCpuKernel;
import backend.cpu.execution.CpuKernelContext;
import operations.Operation;
import tensor.Tensor;

import java.util.List;

public final class CpuEqualToKernel extends TypedCpuKernel implements CompareElementwiseKernel {
    @Override
    protected void forwardBOOL(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        CompareExecutor.execute(this, inputs, node, context);
    }

    @Override
    public boolean testF64(double left, double right) {
        return left == right;
    }

    @Override
    public boolean testF32(float left, float right) {
        return left == right;
    }

    @Override
    public boolean testBF16(float left, float right) {
        return left == right;
    }
}
