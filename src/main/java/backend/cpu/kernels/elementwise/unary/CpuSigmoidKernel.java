package backend.cpu.kernels.elementwise.unary;

import backend.cpu.kernels.TypedCpuKernel;
import backend.cpu.execution.CpuKernelContext;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import operations.Operation;
import tensor.Tensor;

import java.lang.foreign.MemorySegment;
import java.util.List;

import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;

public final class CpuSigmoidKernel extends TypedCpuKernel implements UnaryElementwiseKernel {
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
        return 1.0d / (1.0d + Math.exp(-value));
    }

    @Override
    public float applyF32(float value) {
        return 1.0f / (1.0f + (float) Math.exp(-value));
    }

    @Override
    public float applyBF16(float value) {
        return 1.0f / (1.0f + (float) Math.exp(-value));
    }

    @Override
    public boolean supportsVectorF64() {
        return true;
    }

    @Override
    public DoubleVector applyVectorF64(DoubleVector value) {
        DoubleVector half = DoubleVector.broadcast(value.species(), 0.5d);
        DoubleVector one = DoubleVector.broadcast(value.species(), 1.0d);
        return value.mul(half).lanewise(VectorOperators.TANH).add(one).mul(half);
    }

    @Override
    public boolean supportsVectorF32() {
        return true;
    }

    @Override
    public FloatVector applyVectorF32(FloatVector value) {
        FloatVector half = FloatVector.broadcast(value.species(), 0.5f);
        FloatVector one = FloatVector.broadcast(value.species(), 1.0f);
        return value.mul(half).lanewise(VectorOperators.TANH).add(one).mul(half);
    }

    @Override
    public void runSegmentF64(MemorySegment in, MemorySegment out, int start, int end) {
        for (int i = start; i < end; i++) {
            long offset = (long) i * Double.BYTES;
            double value = in.get(JAVA_DOUBLE, offset);
            out.set(JAVA_DOUBLE, offset, 1.0d / (1.0d + Math.exp(-value)));
        }
    }

    @Override
    public void runSegmentF32(MemorySegment in, MemorySegment out, int start, int end) {
        for (int i = start; i < end; i++) {
            long offset = (long) i * Float.BYTES;
            float value = in.get(JAVA_FLOAT, offset);
            out.set(JAVA_FLOAT, offset, 1.0f / (1.0f + (float) Math.exp(-value)));
        }
    }
}
