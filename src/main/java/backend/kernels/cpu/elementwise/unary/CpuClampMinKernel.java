package backend.kernels.cpu.elementwise.unary;

import backend.kernels.cpu.CpuKernel;
import backend.kernels.cpu.CpuKernelContext;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import operations.Operation;
import operations.elementwise.unary.clampMin;
import tensor.Tensor;

import java.util.List;

public final class CpuClampMinKernel implements CpuKernel, ScalarUnaryElementwiseKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        clampMin clamp = (clampMin) op;
        ElementwiseUnaryExecutor.execute(this, clamp.getMinValue(), clamp.getMinValueF32(), inputs, node, context);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        clampMin clamp = (clampMin) op;
        ElementwiseUnaryExecutor.execute(this, clamp.getMinValue(), clamp.getMinValueF32(), inputs, node, context);
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        clampMin clamp = (clampMin) op;
        ElementwiseUnaryExecutor.execute(this, clamp.getMinValue(), clamp.getMinValueF32(), inputs, node, context);
    }

    @Override
    public double applyF64(double value, double parameter) {
        return Math.max(parameter, value);
    }

    @Override
    public float applyF32(float value, float parameter) {
        return Math.max(parameter, value);
    }

    @Override
    public float applyBF16(float value, float parameter) {
        return Math.max(parameter, value);
    }

    @Override
    public boolean supportsVectorF64() {
        return true;
    }

    @Override
    public DoubleVector applyVectorF64(DoubleVector value, DoubleVector parameter) {
        return value.max(parameter);
    }

    @Override
    public boolean supportsVectorF32() {
        return true;
    }

    @Override
    public FloatVector applyVectorF32(FloatVector value, FloatVector parameter) {
        return value.max(parameter);
    }
}
