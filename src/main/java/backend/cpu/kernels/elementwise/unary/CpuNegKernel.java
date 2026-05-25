package backend.cpu.kernels.elementwise.unary;

import backend.cpu.kernels.TypedCpuKernel;
import backend.cpu.execution.CpuKernelContext;
import tensor.dtype.TensorDTypeOps;
import backend.cpu.kernels.elementwise.ElementwiseRangeLoop;
import backend.cpu.plan.elementwise.ResolvedDispatchHints;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import operations.Operation;
import tensor.Tensor;

import java.lang.foreign.MemorySegment;
import java.util.List;

import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

public final class CpuNegKernel extends TypedCpuKernel implements UnaryElementwiseKernel {
    private static final VectorSpecies<Double> F64 = DoubleVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Float> F32 = FloatVector.SPECIES_PREFERRED;

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
        ElementwiseRangeLoop.run(out.length, hints, true,
                (start, end) -> runArrayF64(in, out, start, end),
                (start, end) -> runArrayVectorF64(in, out, start, end));
    }

    @Override
    public boolean supportsDirectF32() {
        return true;
    }

    @Override
    public void runDirectF32(float[] in, float[] out, ResolvedDispatchHints hints) {
        ElementwiseRangeLoop.run(out.length, hints, true,
                (start, end) -> runArrayF32(in, out, start, end),
                (start, end) -> runArrayVectorF32(in, out, start, end));
    }

    @Override
    public boolean supportsDirectBF16() {
        return true;
    }

    @Override
    public void runDirectBF16(short[] in, float[] continuation, short[] out, ResolvedDispatchHints hints) {
        if (continuation != null) {
            ElementwiseRangeLoop.runScalar(out.length, hints, (start, end) -> runArrayBF16(continuation, out, start, end));
        } else {
            ElementwiseRangeLoop.runScalar(out.length, hints, (start, end) -> runArrayBF16(in, out, start, end));
        }
    }

    private static void runArrayF64(double[] in, double[] out, int start, int end) {
        for (int i = start; i < end; i++) out[i] = -in[i];
    }

    private static void runArrayVectorF64(double[] in, double[] out, int start, int end) {
        int width = F64.length();
        int upper = end - ((end - start) % width);
        int i = start;
        for (; i < upper; i += width) {
            DoubleVector.fromArray(F64, in, i).neg().intoArray(out, i);
        }
        runArrayF64(in, out, i, end);
    }

    private static void runArrayF32(float[] in, float[] out, int start, int end) {
        for (int i = start; i < end; i++) out[i] = -in[i];
    }

    private static void runArrayVectorF32(float[] in, float[] out, int start, int end) {
        int width = F32.length();
        int upper = end - ((end - start) % width);
        int i = start;
        for (; i < upper; i += width) {
            FloatVector.fromArray(F32, in, i).neg().intoArray(out, i);
        }
        runArrayF32(in, out, i, end);
    }

    private static void runArrayBF16(float[] in, short[] out, int start, int end) {
        for (int i = start; i < end; i++) out[i] = TensorDTypeOps.toBFloat16Bits(-in[i]);
    }

    private static void runArrayBF16(short[] in, short[] out, int start, int end) {
        for (int i = start; i < end; i++) out[i] = TensorDTypeOps.toBFloat16Bits(-TensorDTypeOps.fromBFloat16Bits(in[i]));
    }

    @Override
    public void runSegmentF64(MemorySegment in, MemorySegment out, int start, int end) {
        for (int i = start; i < end; i++) {
            long offset = (long) i * Double.BYTES;
            out.set(JAVA_DOUBLE, offset, -in.get(JAVA_DOUBLE, offset));
        }
    }

    @Override
    public void runSegmentF32(MemorySegment in, MemorySegment out, int start, int end) {
        for (int i = start; i < end; i++) {
            long offset = (long) i * Float.BYTES;
            out.set(JAVA_FLOAT, offset, -in.get(JAVA_FLOAT, offset));
        }
    }

    @Override
    public void runSegmentBF16(MemorySegment in, MemorySegment out, int start, int end) {
        for (int i = start; i < end; i++) {
            long offset = (long) i * Short.BYTES;
            float value = TensorDTypeOps.fromBFloat16Bits(in.get(JAVA_SHORT, offset));
            out.set(JAVA_SHORT, offset, TensorDTypeOps.toBFloat16Bits(-value));
        }
    }
}
