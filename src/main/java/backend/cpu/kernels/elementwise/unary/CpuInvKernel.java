package backend.cpu.kernels.elementwise.unary;

import backend.cpu.kernels.CpuKernel;
import backend.cpu.kernels.CpuKernelContext;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import operations.Operation;
import tensor.Tensor;

import java.util.List;

public final class CpuInvKernel implements CpuKernel, UnaryElementwiseKernel {
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
        return 1.0d / value;
    }

    @Override
    public float applyF32(float value) {
        return 1.0f / value;
    }

    @Override
    public float applyBF16(float value) {
        return 1.0f / value;
    }

    @Override
    public boolean supportsVectorF64() {
        return true;
    }

    @Override
    public DoubleVector applyVectorF64(DoubleVector value) {
        return DoubleVector.broadcast(value.species(), 1.0d).div(value);
    }

    @Override
    public boolean supportsVectorF32() {
        return true;
    }

    @Override
    public FloatVector applyVectorF32(FloatVector value) {
        return FloatVector.broadcast(value.species(), 1.0f).div(value);
    }
}
