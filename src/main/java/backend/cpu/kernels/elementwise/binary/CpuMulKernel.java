package backend.cpu.kernels.elementwise.binary;

import backend.cpu.kernels.CpuKernel;
import backend.cpu.kernels.CpuKernelContext;
import backend.cpu.kernels.elementwise.plan.ResolvedDispatchHints;
import backend.cpu.kernels.elementwise.binary.array.MulBF16;
import backend.cpu.kernels.elementwise.binary.array.MulF32;
import backend.cpu.kernels.elementwise.binary.array.MulF64;
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

    @Override
    public boolean supportsDirectF64() {
        return true;
    }

    @Override
    public void runDirectF64(double[] left, double[] right, double[] out, ResolvedDispatchHints hints) {
        MulF64.run(left, right, out, hints);
    }

    @Override
    public boolean supportsDirectF32() {
        return true;
    }

    @Override
    public void runDirectF32(float[] left, float[] right, float[] out, ResolvedDispatchHints hints) {
        MulF32.run(left, right, out, hints);
    }

    @Override
    public boolean supportsDirectBF16() {
        return true;
    }

    @Override
    public void runDirectBF16(
            short[] leftStorage,
            short[] rightStorage,
            float[] leftContinuation,
            float[] rightContinuation,
            short[] out,
            ResolvedDispatchHints hints
    ) {
        if (leftContinuation != null && rightContinuation != null) {
            MulBF16.run(leftContinuation, rightContinuation, out, hints);
        } else if (leftContinuation != null) {
            MulBF16.run(leftContinuation, rightStorage, out, hints);
        } else if (rightContinuation != null) {
            MulBF16.run(leftStorage, rightContinuation, out, hints);
        } else {
            MulBF16.run(leftStorage, rightStorage, out, hints);
        }
    }
}
