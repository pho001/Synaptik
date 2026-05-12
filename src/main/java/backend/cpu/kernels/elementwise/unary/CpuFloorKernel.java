package backend.cpu.kernels.elementwise.unary;

import backend.cpu.kernels.CpuKernel;
import backend.cpu.kernels.CpuKernelContext;
import operations.Operation;
import tensor.Tensor;

import java.util.List;

public final class CpuFloorKernel implements CpuKernel, UnaryElementwiseKernel {
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
        return Math.floor(value);
    }

    @Override
    public float applyF32(float value) {
        return (float) Math.floor(value);
    }

    @Override
    public float applyBF16(float value) {
        return (float) Math.floor(value);
    }

}
