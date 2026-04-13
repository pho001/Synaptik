package backend.kernels.cpu.elementwise.unary;

import backend.kernels.cpu.CpuKernel;
import backend.kernels.cpu.CpuKernelContext;
import operations.Operation;
import operations.pow;
import tensor.Tensor;

import java.util.List;

public final class CpuPowKernel implements CpuKernel, ScalarUnaryElementwiseKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        pow power = (pow) op;
        ElementwiseUnaryExecutor.execute(this, power.getExponent(), power.getExponentF32(), inputs, node, context);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        pow power = (pow) op;
        ElementwiseUnaryExecutor.execute(this, power.getExponent(), power.getExponentF32(), inputs, node, context);
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        pow power = (pow) op;
        ElementwiseUnaryExecutor.execute(this, power.getExponent(), power.getExponentF32(), inputs, node, context);
    }

    @Override
    public double applyF64(double value, double parameter) {
        return Math.pow(value, parameter);
    }

    @Override
    public float applyF32(float value, float parameter) {
        return (float) Math.pow(value, parameter);
    }

    @Override
    public float applyBF16(float value, float parameter) {
        return (float) Math.pow(value, parameter);
    }
}
