package backend.cpu.kernels.elementwise.unary;

import backend.cpu.kernels.elementwise.ElementwiseOffsetCursor;
import backend.cpu.kernels.elementwise.ElementwiseRangeLoop;
import backend.cpu.plan.elementwise.ResolvedDispatchHints;
import backend.cpu.storage.CpuStorageView;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import operations.Operation;
import operations.elementwise.unary.clampMax;
import tensor.dtype.TensorDTypeOps;

import java.lang.foreign.MemorySegment;

import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

public final class CpuClampMaxKernel extends StorageAwareScalarUnaryElementwiseKernel {
    private static final VectorSpecies<Double> F64 = DoubleVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Float> F32 = FloatVector.SPECIES_PREFERRED;

    @Override
    protected Operation.OpType opType() {
        return Operation.OpType.CLAMP_MAX;
    }

    @Override
    protected String opLabel() {
        return "clamp_max";
    }

    @Override
    protected ScalarUnaryParameters extractParameters(Operation operation) {
        clampMax clamp = (clampMax) operation;
        return new ScalarUnaryParameters(clamp.getMaxValue(), clamp.getMaxValueF32());
    }

    @Override
    protected void runDirectF64(double[] in, double parameter, double[] out, ResolvedDispatchHints hints) {
        ElementwiseRangeLoop.run(out.length, hints, true,
                (start, end) -> runArrayF64(in, parameter, out, start, end),
                (start, end) -> runArrayVectorF64(in, parameter, out, start, end));
    }

    @Override
    protected void runDirectF32(float[] in, float parameter, float[] out, ResolvedDispatchHints hints) {
        ElementwiseRangeLoop.run(out.length, hints, true,
                (start, end) -> runArrayF32(in, parameter, out, start, end),
                (start, end) -> runArrayVectorF32(in, parameter, out, start, end));
    }

    @Override
    protected void runDirectBF16(short[] in, float[] continuation, float parameter, short[] out, ResolvedDispatchHints hints) {
        if (continuation != null) {
            ElementwiseRangeLoop.runScalar(out.length, hints,
                    (start, end) -> runArrayBF16(continuation, parameter, out, start, end));
        } else {
            ElementwiseRangeLoop.runScalar(out.length, hints,
                    (start, end) -> runArrayBF16(in, parameter, out, start, end));
        }
    }

    @Override
    protected void runDirectBF16ToFloat(
            short[] in,
            float[] continuation,
            float parameter,
            float[] out,
            ResolvedDispatchHints hints
    ) {
        if (continuation != null) {
            runDirectF32(continuation, parameter, out, hints);
            return;
        }
        ElementwiseRangeLoop.runScalar(out.length, hints,
                (start, end) -> runArrayBF16ToFloat(in, parameter, out, start, end));
    }

    private static void runArrayF64(double[] in, double parameter, double[] out, int start, int end) {
        for (int i = start; i < end; i++) out[i] = Math.min(parameter, in[i]);
    }

    private static void runArrayVectorF64(double[] in, double parameter, double[] out, int start, int end) {
        int width = F64.length();
        int upper = end - ((end - start) % width);
        DoubleVector parameterVector = DoubleVector.broadcast(F64, parameter);
        int i = start;
        for (; i < upper; i += width) {
            DoubleVector.fromArray(F64, in, i).min(parameterVector).intoArray(out, i);
        }
        runArrayF64(in, parameter, out, i, end);
    }

    private static void runArrayF32(float[] in, float parameter, float[] out, int start, int end) {
        for (int i = start; i < end; i++) out[i] = Math.min(parameter, in[i]);
    }

    private static void runArrayVectorF32(float[] in, float parameter, float[] out, int start, int end) {
        int width = F32.length();
        int upper = end - ((end - start) % width);
        FloatVector parameterVector = FloatVector.broadcast(F32, parameter);
        int i = start;
        for (; i < upper; i += width) {
            FloatVector.fromArray(F32, in, i).min(parameterVector).intoArray(out, i);
        }
        runArrayF32(in, parameter, out, i, end);
    }

    private static void runArrayBF16(float[] in, float parameter, short[] out, int start, int end) {
        for (int i = start; i < end; i++) out[i] = TensorDTypeOps.toBFloat16Bits(Math.min(parameter, in[i]));
    }

    private static void runArrayBF16(short[] in, float parameter, short[] out, int start, int end) {
        for (int i = start; i < end; i++) {
            out[i] = TensorDTypeOps.toBFloat16Bits(Math.min(parameter, TensorDTypeOps.fromBFloat16Bits(in[i])));
        }
    }

    private static void runArrayBF16ToFloat(short[] in, float parameter, float[] out, int start, int end) {
        for (int i = start; i < end; i++) out[i] = Math.min(parameter, TensorDTypeOps.fromBFloat16Bits(in[i]));
    }

    @Override
    protected void runSegmentF64(MemorySegment in, double parameter, MemorySegment out, int start, int end) {
        for (int i = start; i < end; i++) {
            long offset = (long) i * Double.BYTES;
            out.set(JAVA_DOUBLE, offset, Math.min(parameter, in.get(JAVA_DOUBLE, offset)));
        }
    }

    @Override
    protected void runSegmentF32(MemorySegment in, float parameter, MemorySegment out, int start, int end) {
        for (int i = start; i < end; i++) {
            long offset = (long) i * Float.BYTES;
            out.set(JAVA_FLOAT, offset, Math.min(parameter, in.get(JAVA_FLOAT, offset)));
        }
    }

    @Override
    protected void runSegmentBF16(MemorySegment in, float parameter, MemorySegment out, int start, int end) {
        for (int i = start; i < end; i++) {
            long offset = (long) i * Short.BYTES;
            float value = TensorDTypeOps.fromBFloat16Bits(in.get(JAVA_SHORT, offset));
            out.set(JAVA_SHORT, offset, TensorDTypeOps.toBFloat16Bits(Math.min(parameter, value)));
        }
    }

    @Override
    protected void runIndexedArrayF64(
            double[] in,
            double parameter,
            double[] out,
            ScalarUnaryStorageLayout layout,
            int start,
            int end
    ) {
        ElementwiseOffsetCursor cursor = layout.cursor(start);
        for (int outIndex = start; outIndex < end; outIndex++) {
            out[cursor.offset(0)] = Math.min(parameter, in[cursor.offset(1)]);
            if (outIndex + 1 < end) {
                cursor.step();
            }
        }
    }

    @Override
    protected void runIndexedArrayF32(
            float[] in,
            float parameter,
            float[] out,
            ScalarUnaryStorageLayout layout,
            int start,
            int end
    ) {
        ElementwiseOffsetCursor cursor = layout.cursor(start);
        for (int outIndex = start; outIndex < end; outIndex++) {
            out[cursor.offset(0)] = Math.min(parameter, in[cursor.offset(1)]);
            if (outIndex + 1 < end) {
                cursor.step();
            }
        }
    }

    @Override
    protected void runIndexedArrayBF16(
            short[] in,
            float[] continuation,
            float parameter,
            short[] out,
            ScalarUnaryStorageLayout layout,
            int start,
            int end
    ) {
        ElementwiseOffsetCursor cursor = layout.cursor(start);
        for (int outIndex = start; outIndex < end; outIndex++) {
            out[cursor.offset(0)] = TensorDTypeOps.toBFloat16Bits(
                    Math.min(parameter, loadBF16(continuation, in, cursor.offset(1)))
            );
            if (outIndex + 1 < end) {
                cursor.step();
            }
        }
    }

    @Override
    protected void runIndexedArrayBF16ToFloat(
            short[] in,
            float[] continuation,
            float parameter,
            float[] out,
            ScalarUnaryStorageLayout layout,
            int start,
            int end
    ) {
        ElementwiseOffsetCursor cursor = layout.cursor(start);
        for (int outIndex = start; outIndex < end; outIndex++) {
            out[outIndex] = Math.min(parameter, loadBF16(continuation, in, cursor.offset(1)));
            if (outIndex + 1 < end) {
                cursor.step();
            }
        }
    }

    @Override
    protected void runIndexedSegmentF64(
            MemorySegment in,
            double parameter,
            MemorySegment out,
            ScalarUnaryStorageLayout layout,
            int start,
            int end
    ) {
        ElementwiseOffsetCursor cursor = layout.cursor(start);
        for (int outIndex = start; outIndex < end; outIndex++) {
            out.set(
                    JAVA_DOUBLE,
                    (long) cursor.offset(0) * Double.BYTES,
                    Math.min(parameter, in.get(JAVA_DOUBLE, (long) cursor.offset(1) * Double.BYTES))
            );
            if (outIndex + 1 < end) {
                cursor.step();
            }
        }
    }

    @Override
    protected void runIndexedSegmentF32(
            MemorySegment in,
            float parameter,
            MemorySegment out,
            ScalarUnaryStorageLayout layout,
            int start,
            int end
    ) {
        ElementwiseOffsetCursor cursor = layout.cursor(start);
        for (int outIndex = start; outIndex < end; outIndex++) {
            out.set(
                    JAVA_FLOAT,
                    (long) cursor.offset(0) * Float.BYTES,
                    Math.min(parameter, in.get(JAVA_FLOAT, (long) cursor.offset(1) * Float.BYTES))
            );
            if (outIndex + 1 < end) {
                cursor.step();
            }
        }
    }

    @Override
    protected void runIndexedSegmentBF16(
            MemorySegment in,
            float parameter,
            MemorySegment out,
            ScalarUnaryStorageLayout layout,
            int start,
            int end
    ) {
        ElementwiseOffsetCursor cursor = layout.cursor(start);
        for (int outIndex = start; outIndex < end; outIndex++) {
            float value = TensorDTypeOps.fromBFloat16Bits(
                    in.get(JAVA_SHORT, (long) cursor.offset(1) * Short.BYTES)
            );
            out.set(
                    JAVA_SHORT,
                    (long) cursor.offset(0) * Short.BYTES,
                    TensorDTypeOps.toBFloat16Bits(Math.min(parameter, value))
            );
            if (outIndex + 1 < end) {
                cursor.step();
            }
        }
    }

    @Override
    protected void runIndexedMixedF64(
            CpuStorageView in,
            double parameter,
            CpuStorageView out,
            ScalarUnaryStorageLayout layout,
            int start,
            int end
    ) {
        ElementwiseOffsetCursor cursor = layout.cursor(start);
        for (int outIndex = start; outIndex < end; outIndex++) {
            writeF64(out, cursor.offset(0), Math.min(parameter, readF64(in, cursor.offset(1))));
            if (outIndex + 1 < end) {
                cursor.step();
            }
        }
    }

    @Override
    protected void runIndexedMixedF32(
            CpuStorageView in,
            float parameter,
            CpuStorageView out,
            ScalarUnaryStorageLayout layout,
            int start,
            int end
    ) {
        ElementwiseOffsetCursor cursor = layout.cursor(start);
        for (int outIndex = start; outIndex < end; outIndex++) {
            writeF32(out, cursor.offset(0), Math.min(parameter, readF32(in, cursor.offset(1))));
            if (outIndex + 1 < end) {
                cursor.step();
            }
        }
    }

    @Override
    protected void runIndexedMixedBF16(
            CpuStorageView in,
            float parameter,
            CpuStorageView out,
            ScalarUnaryStorageLayout layout,
            int start,
            int end
    ) {
        ElementwiseOffsetCursor cursor = layout.cursor(start);
        for (int outIndex = start; outIndex < end; outIndex++) {
            writeBF16(out, cursor.offset(0), Math.min(parameter, readBF16(in, cursor.offset(1))));
            if (outIndex + 1 < end) {
                cursor.step();
            }
        }
    }
}
