package backend.kernels.cpu.elementwise.unary;

import backend.kernels.cpu.CpuKernel;
import backend.kernels.cpu.CpuKernelContext;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import operations.Operation;
import tensor.Tensor;

import java.util.List;

public final class CpuLogKernel implements CpuKernel, UnaryElementwiseKernel {
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
        return Math.log(value);
    }

    @Override
    public float applyF32(float value) {
        return (float) Math.log(value);
    }

    @Override
    public float applyBF16(float value) {
        return (float) Math.log(value);
    }

    @Override
    public boolean supportsVectorF64() {
        return true;
    }

    @Override
    public DoubleVector applyVectorF64(DoubleVector value) {
        return value.lanewise(VectorOperators.LOG);
    }

    @Override
    public boolean supportsVectorF32() {
        return true;
    }

    @Override
    public FloatVector applyVectorF32(FloatVector value) {
        return value.lanewise(VectorOperators.LOG);
    }
}
