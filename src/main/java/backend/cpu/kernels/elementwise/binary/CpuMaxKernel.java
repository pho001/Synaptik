package backend.cpu.kernels.elementwise.binary;

import backend.cpu.kernels.TypedCpuKernel;
import backend.cpu.execution.CpuKernelContext;
import tensor.dtype.TensorDTypeOps;
import backend.cpu.kernels.elementwise.ElementwiseRangeLoop;
import backend.cpu.plan.elementwise.ResolvedDispatchHints;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import jdk.incubator.vector.VectorOperators;
import operations.Operation;
import tensor.Tensor;

import java.lang.foreign.MemorySegment;
import java.util.List;

import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

public final class CpuMaxKernel extends TypedCpuKernel implements BinaryElementwiseKernel {
    private static final VectorSpecies<Double> F64 = DoubleVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Float> F32 = FloatVector.SPECIES_PREFERRED;

    @Override
    protected void forwardF64(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        ElementwiseBinaryExecutor.execute(this, inputs, node, context);
    }

    @Override
    protected void forwardF32(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        ElementwiseBinaryExecutor.execute(this, inputs, node, context);
    }

    @Override
    protected void forwardBF16(Operation op, List<Tensor> inputs, Tensor node, CpuKernelContext context) {
        ElementwiseBinaryExecutor.execute(this, inputs, node, context);
    }

    @Override
    public double applyF64(double left, double right) {
        return Math.max(left, right);
    }

    @Override
    public float applyF32(float left, float right) {
        return Math.max(left, right);
    }

    @Override
    public float applyBF16(float left, float right) {
        return Math.max(left, right);
    }

    @Override
    public boolean supportsVectorF64() {
        return true;
    }

    @Override
    public DoubleVector applyVectorF64(DoubleVector left, DoubleVector right) {
        return left.lanewise(VectorOperators.MAX, right);
    }

    @Override
    public boolean supportsVectorF32() {
        return true;
    }

    @Override
    public FloatVector applyVectorF32(FloatVector left, FloatVector right) {
        return left.lanewise(VectorOperators.MAX, right);
    }

    @Override
    public boolean supportsDirectF64() {
        return true;
    }

    @Override
    public void runDirectF64(double[] left, double[] right, double[] out, ResolvedDispatchHints hints) {
        ElementwiseRangeLoop.run(out.length, hints, true,
                (start, end) -> runArrayF64(left, right, out, start, end),
                (start, end) -> runArrayVectorF64(left, right, out, start, end));
    }

    @Override
    public boolean supportsDirectF32() {
        return true;
    }

    @Override
    public void runDirectF32(float[] left, float[] right, float[] out, ResolvedDispatchHints hints) {
        ElementwiseRangeLoop.run(out.length, hints, true,
                (start, end) -> runArrayF32(left, right, out, start, end),
                (start, end) -> runArrayVectorF32(left, right, out, start, end));
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
            ElementwiseRangeLoop.runScalar(out.length, hints, (start, end) -> runArrayBF16(leftContinuation, rightContinuation, out, start, end));
        } else if (leftContinuation != null) {
            ElementwiseRangeLoop.runScalar(out.length, hints, (start, end) -> runArrayBF16(leftContinuation, rightStorage, out, start, end));
        } else if (rightContinuation != null) {
            ElementwiseRangeLoop.runScalar(out.length, hints, (start, end) -> runArrayBF16(leftStorage, rightContinuation, out, start, end));
        } else {
            ElementwiseRangeLoop.runScalar(out.length, hints, (start, end) -> runArrayBF16(leftStorage, rightStorage, out, start, end));
        }
    }

    private static void runArrayF64(double[] left, double[] right, double[] out, int start, int end) {
        for (int i = start; i < end; i++) out[i] = Math.max(left[i], right[i]);
    }

    private static void runArrayVectorF64(double[] left, double[] right, double[] out, int start, int end) {
        int width = F64.length();
        int upper = end - ((end - start) % width);
        int i = start;
        for (; i < upper; i += width) {
            DoubleVector.fromArray(F64, left, i).lanewise(VectorOperators.MAX, DoubleVector.fromArray(F64, right, i)).intoArray(out, i);
        }
        runArrayF64(left, right, out, i, end);
    }

    private static void runArrayF32(float[] left, float[] right, float[] out, int start, int end) {
        for (int i = start; i < end; i++) out[i] = Math.max(left[i], right[i]);
    }

    private static void runArrayVectorF32(float[] left, float[] right, float[] out, int start, int end) {
        int width = F32.length();
        int upper = end - ((end - start) % width);
        int i = start;
        for (; i < upper; i += width) {
            FloatVector.fromArray(F32, left, i).lanewise(VectorOperators.MAX, FloatVector.fromArray(F32, right, i)).intoArray(out, i);
        }
        runArrayF32(left, right, out, i, end);
    }

    private static void runArrayBF16(float[] left, float[] right, short[] out, int start, int end) {
        for (int i = start; i < end; i++) out[i] = TensorDTypeOps.toBFloat16Bits(Math.max(left[i], right[i]));
    }

    private static void runArrayBF16(float[] left, short[] right, short[] out, int start, int end) {
        for (int i = start; i < end; i++) out[i] = TensorDTypeOps.toBFloat16Bits(Math.max(left[i], TensorDTypeOps.fromBFloat16Bits(right[i])));
    }

    private static void runArrayBF16(short[] left, float[] right, short[] out, int start, int end) {
        for (int i = start; i < end; i++) out[i] = TensorDTypeOps.toBFloat16Bits(Math.max(TensorDTypeOps.fromBFloat16Bits(left[i]), right[i]));
    }

    private static void runArrayBF16(short[] left, short[] right, short[] out, int start, int end) {
        for (int i = start; i < end; i++) {
            float leftValue = TensorDTypeOps.fromBFloat16Bits(left[i]);
            float rightValue = TensorDTypeOps.fromBFloat16Bits(right[i]);
            out[i] = TensorDTypeOps.toBFloat16Bits(Math.max(leftValue, rightValue));
        }
    }

    @Override
    public void runSegmentF64(MemorySegment left, MemorySegment right, MemorySegment out, int start, int end) {
        for (int i = start; i < end; i++) {
            long offset = (long) i * Double.BYTES;
            out.set(JAVA_DOUBLE, offset, Math.max(left.get(JAVA_DOUBLE, offset), right.get(JAVA_DOUBLE, offset)));
        }
    }

    @Override
    public void runSegmentF32(MemorySegment left, MemorySegment right, MemorySegment out, int start, int end) {
        for (int i = start; i < end; i++) {
            long offset = (long) i * Float.BYTES;
            out.set(JAVA_FLOAT, offset, Math.max(left.get(JAVA_FLOAT, offset), right.get(JAVA_FLOAT, offset)));
        }
    }

    @Override
    public void runSegmentBF16(MemorySegment left, MemorySegment right, MemorySegment out, int start, int end) {
        for (int i = start; i < end; i++) {
            long offset = (long) i * Short.BYTES;
            float leftValue = TensorDTypeOps.fromBFloat16Bits(left.get(JAVA_SHORT, offset));
            float rightValue = TensorDTypeOps.fromBFloat16Bits(right.get(JAVA_SHORT, offset));
            out.set(JAVA_SHORT, offset, TensorDTypeOps.toBFloat16Bits(Math.max(leftValue, rightValue)));
        }
    }
}
