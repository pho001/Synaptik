package backend.cpu.kernels.elementwise.unary;

import backend.cpu.kernels.TypedCpuKernel;
import backend.cpu.execution.CpuKernelContext;
import operations.Operation;
import tensor.Tensor;
import utils.SpecialFunctions;

import java.util.List;

public final class CpuErfKernel extends TypedCpuKernel implements UnaryElementwiseKernel {
    @Override
    protected void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        ElementwiseUnaryExecutor.execute(this, inputs, node, context);
    }

    @Override
    protected void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        ElementwiseUnaryExecutor.execute(this, inputs, node, context);
    }

    @Override
    protected void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        ElementwiseUnaryExecutor.execute(this, inputs, node, context);
    }

    @Override
    public double applyF64(double value) {
        return SpecialFunctions.erf(value);
    }

    @Override
    public float applyF32(float value) {
        return SpecialFunctions.erf(value);
    }

    @Override
    public float applyBF16(float value) {
        return SpecialFunctions.erf(value);
    }
}
