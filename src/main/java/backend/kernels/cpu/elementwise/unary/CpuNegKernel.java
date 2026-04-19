package backend.kernels.cpu.elementwise.unary;

import backend.kernels.cpu.CpuKernel;
import backend.kernels.cpu.CpuKernelContext;
import backend.kernels.cpu.elementwise.plan.ResolvedDispatchHints;
import backend.kernels.cpu.elementwise.unary.bf16.NegBF16;
import backend.kernels.cpu.elementwise.unary.f32.NegF32;
import backend.kernels.cpu.elementwise.unary.f64.NegF64;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import operations.Operation;
import tensor.Tensor;

import java.util.List;

public final class CpuNegKernel implements CpuKernel, UnaryElementwiseKernel {
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
        return -value;
    }

    @Override
    public float applyF32(float value) {
        return -value;
    }

    @Override
    public float applyBF16(float value) {
        return -value;
    }

    @Override
    public boolean supportsVectorF64() {
        return true;
    }

    @Override
    public DoubleVector applyVectorF64(DoubleVector value) {
        return value.neg();
    }

    @Override
    public boolean supportsVectorF32() {
        return true;
    }

    @Override
    public FloatVector applyVectorF32(FloatVector value) {
        return value.neg();
    }

    @Override
    public boolean supportsDirectF64() {
        return true;
    }

    @Override
    public void runDirectF64(double[] in, double[] out, ResolvedDispatchHints hints) {
        NegF64.run(in, out, hints);
    }

    @Override
    public boolean supportsDirectF32() {
        return true;
    }

    @Override
    public void runDirectF32(float[] in, float[] out, ResolvedDispatchHints hints) {
        NegF32.run(in, out, hints);
    }

    @Override
    public boolean supportsDirectBF16() {
        return true;
    }

    @Override
    public void runDirectBF16(short[] in, float[] continuation, short[] out, ResolvedDispatchHints hints) {
        if (continuation != null) {
            NegBF16.run(continuation, out, hints);
        } else {
            NegBF16.run(in, out, hints);
        }
    }
}
