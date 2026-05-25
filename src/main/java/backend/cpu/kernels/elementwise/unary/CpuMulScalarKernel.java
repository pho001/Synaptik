package backend.cpu.kernels.elementwise.unary;

import backend.cpu.kernels.CpuKernel;
import backend.cpu.kernels.CpuKernelContext;
import backend.cpu.kernels.elementwise.plan.ResolvedDispatchHints;
import backend.cpu.kernels.elementwise.unary.array.MulScalarBF16;
import backend.cpu.kernels.elementwise.unary.array.MulScalarF32;
import backend.cpu.kernels.elementwise.unary.array.MulScalarF64;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import operations.Operation;
import operations.elementwise.unary.mulScalar;
import tensor.Tensor;

import java.util.List;

public final class CpuMulScalarKernel implements CpuKernel, ScalarUnaryElementwiseKernel {
    @Override
    public void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        mulScalar mul = (mulScalar) op;
        ElementwiseUnaryExecutor.execute(this, mul.getScalar(), mul.getScalarF32(), inputs, node, context);
    }

    @Override
    public void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        mulScalar mul = (mulScalar) op;
        ElementwiseUnaryExecutor.execute(this, mul.getScalar(), mul.getScalarF32(), inputs, node, context);
    }

    @Override
    public void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        mulScalar mul = (mulScalar) op;
        ElementwiseUnaryExecutor.execute(this, mul.getScalar(), mul.getScalarF32(), inputs, node, context);
    }

    @Override
    public double applyF64(double value, double parameter) {
        return value * parameter;
    }

    @Override
    public float applyF32(float value, float parameter) {
        return value * parameter;
    }

    @Override
    public float applyBF16(float value, float parameter) {
        return value * parameter;
    }

    @Override
    public boolean supportsVectorF64() {
        return true;
    }

    @Override
    public DoubleVector applyVectorF64(DoubleVector value, DoubleVector parameter) {
        return value.mul(parameter);
    }

    @Override
    public boolean supportsVectorF32() {
        return true;
    }

    @Override
    public FloatVector applyVectorF32(FloatVector value, FloatVector parameter) {
        return value.mul(parameter);
    }

    @Override
    public boolean supportsDirectF64() {
        return true;
    }

    @Override
    public void runDirectF64(double[] in, double parameter, double[] out, ResolvedDispatchHints hints) {
        MulScalarF64.run(in, parameter, out, hints);
    }

    @Override
    public boolean supportsDirectF32() {
        return true;
    }

    @Override
    public void runDirectF32(float[] in, float parameter, float[] out, ResolvedDispatchHints hints) {
        MulScalarF32.run(in, parameter, out, hints);
    }

    @Override
    public boolean supportsDirectBF16() {
        return true;
    }

    @Override
    public void runDirectBF16(short[] in, float[] continuation, float parameter, short[] out, ResolvedDispatchHints hints) {
        if (continuation != null) {
            MulScalarBF16.run(continuation, parameter, out, hints);
        } else {
            MulScalarBF16.run(in, parameter, out, hints);
        }
    }
}
