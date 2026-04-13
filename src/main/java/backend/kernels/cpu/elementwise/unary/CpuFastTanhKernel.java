package backend.kernels.cpu.elementwise.unary;

import backend.kernels.cpu.CpuKernel;
import backend.kernels.cpu.CpuKernelContext;
import operations.Operation;
import tensor.Tensor;
import utils.FastExp;

import java.util.List;

public final class CpuFastTanhKernel implements CpuKernel, UnaryElementwiseKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        ElementwiseUnaryExecutor.execute(this, inputs, node, context);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        ElementwiseUnaryExecutor.execute(this, inputs, node, context);
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        ElementwiseUnaryExecutor.execute(this, inputs, node, context);
    }

    @Override
    public double applyF64(double value) {
        return FastExp.fastTanhF64(value);
    }

    @Override
    public float applyF32(float value) {
        return FastExp.fastTanhF32(value);
    }

    @Override
    public float applyBF16(float value) {
        return FastExp.fastTanhF32(value);
    }
}
