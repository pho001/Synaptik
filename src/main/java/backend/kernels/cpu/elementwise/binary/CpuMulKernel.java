package backend.kernels.cpu.elementwise.binary;

import backend.kernels.cpu.CpuKernel;
import backend.kernels.cpu.CpuKernelContext;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import operations.Operation;
import tensor.Tensor;

import java.util.List;

public final class CpuMulKernel implements CpuKernel, BinaryElementwiseKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        ElementwiseBinaryExecutor.execute(this, inputs, node, context);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        ElementwiseBinaryExecutor.execute(this, inputs, node, context);
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        ElementwiseBinaryExecutor.execute(this, inputs, node, context);
    }

    @Override
    public double applyF64(double left, double right) {
        return left * right;
    }

    @Override
    public float applyF32(float left, float right) {
        return left * right;
    }

    @Override
    public float applyBF16(float left, float right) {
        return left * right;
    }

    @Override
    public boolean supportsVectorF64() {
        return true;
    }

    @Override
    public DoubleVector applyVectorF64(DoubleVector left, DoubleVector right) {
        return left.mul(right);
    }

    @Override
    public boolean supportsVectorF32() {
        return true;
    }

    @Override
    public FloatVector applyVectorF32(FloatVector left, FloatVector right) {
        return left.mul(right);
    }
}
