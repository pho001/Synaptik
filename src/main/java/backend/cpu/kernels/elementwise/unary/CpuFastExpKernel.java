package backend.cpu.kernels.elementwise.unary;

import backend.cpu.kernels.elementwise.ElementwiseOffsetCursor;
import backend.cpu.kernels.elementwise.ElementwiseRangeLoop;
import backend.cpu.plan.elementwise.ResolvedDispatchHints;
import backend.cpu.storage.CpuStorageView;
import operations.Operation;
import tensor.DataType;
import tensor.dtype.TensorDTypeOps;
import utils.FastTranscendentals;

import java.lang.foreign.MemorySegment;

import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

public final class CpuFastExpKernel extends StorageAwareUnaryElementwiseKernel {
    @Override
    protected Operation.OpType opType() {
        return Operation.OpType.FAST_EXP;
    }

    @Override
    protected String opLabel() {
        return "fast_exp";
    }

    @Override
    protected boolean supportsNativeOpDType(DataType dtype) {
        return dtype == DataType.FLOAT32 || dtype == DataType.FLOAT64;
    }

    @Override
    protected void runDirectF64(double[] in, double[] out, ResolvedDispatchHints hints) {
        ElementwiseRangeLoop.runScalar(out.length, hints, (start, end) -> runArrayF64(in, out, start, end));
    }

    @Override
    protected void runDirectF32(float[] in, float[] out, ResolvedDispatchHints hints) {
        ElementwiseRangeLoop.runScalar(out.length, hints, (start, end) -> runArrayF32(in, out, start, end));
    }

    @Override
    protected void runDirectBF16(short[] in, float[] continuation, short[] out, ResolvedDispatchHints hints) {
        if (continuation != null) {
            ElementwiseRangeLoop.runScalar(out.length, hints,
                    (start, end) -> runArrayBF16(continuation, out, start, end));
        } else {
            ElementwiseRangeLoop.runScalar(out.length, hints,
                    (start, end) -> runArrayBF16(in, out, start, end));
        }
    }

    @Override
    protected void runDirectBF16ToFloat(short[] in, float[] continuation, float[] out, ResolvedDispatchHints hints) {
        if (continuation != null) {
            runDirectF32(continuation, out, hints);
            return;
        }
        ElementwiseRangeLoop.runScalar(out.length, hints,
                (start, end) -> runArrayBF16ToFloat(in, out, start, end));
    }

    private static void runArrayF64(double[] in, double[] out, int start, int end) {
        for (int i = start; i < end; i++) out[i] = FastTranscendentals.fastExpF64(in[i]);
    }

    private static void runArrayF32(float[] in, float[] out, int start, int end) {
        for (int i = start; i < end; i++) out[i] = FastTranscendentals.fastExpF32(in[i]);
    }

    private static void runArrayBF16(float[] in, short[] out, int start, int end) {
        for (int i = start; i < end; i++) {
            out[i] = TensorDTypeOps.toBFloat16Bits(FastTranscendentals.fastExpF32(in[i]));
        }
    }

    private static void runArrayBF16(short[] in, short[] out, int start, int end) {
        for (int i = start; i < end; i++) {
            out[i] = TensorDTypeOps.toBFloat16Bits(
                    FastTranscendentals.fastExpF32(TensorDTypeOps.fromBFloat16Bits(in[i]))
            );
        }
    }

    private static void runArrayBF16ToFloat(short[] in, float[] out, int start, int end) {
        for (int i = start; i < end; i++) {
            out[i] = FastTranscendentals.fastExpF32(TensorDTypeOps.fromBFloat16Bits(in[i]));
        }
    }

    @Override
    protected void runSegmentF64(MemorySegment in, MemorySegment out, int start, int end) {
        for (int i = start; i < end; i++) {
            long offset = (long) i * Double.BYTES;
            out.set(JAVA_DOUBLE, offset, FastTranscendentals.fastExpF64(in.get(JAVA_DOUBLE, offset)));
        }
    }

    @Override
    protected void runSegmentF32(MemorySegment in, MemorySegment out, int start, int end) {
        for (int i = start; i < end; i++) {
            long offset = (long) i * Float.BYTES;
            out.set(JAVA_FLOAT, offset, FastTranscendentals.fastExpF32(in.get(JAVA_FLOAT, offset)));
        }
    }

    @Override
    protected void runSegmentBF16(MemorySegment in, MemorySegment out, int start, int end) {
        for (int i = start; i < end; i++) {
            long offset = (long) i * Short.BYTES;
            float value = TensorDTypeOps.fromBFloat16Bits(in.get(JAVA_SHORT, offset));
            out.set(JAVA_SHORT, offset, TensorDTypeOps.toBFloat16Bits(FastTranscendentals.fastExpF32(value)));
        }
    }

    @Override
    protected void runIndexedArrayF64(double[] in, double[] out, UnaryStorageLayout layout, int start, int end) {
        ElementwiseOffsetCursor cursor = layout.cursor(start);
        for (int outIndex = start; outIndex < end; outIndex++) {
            out[cursor.offset(0)] = FastTranscendentals.fastExpF64(in[cursor.offset(1)]);
            if (outIndex + 1 < end) {
                cursor.step();
            }
        }
    }

    @Override
    protected void runIndexedArrayF32(float[] in, float[] out, UnaryStorageLayout layout, int start, int end) {
        ElementwiseOffsetCursor cursor = layout.cursor(start);
        for (int outIndex = start; outIndex < end; outIndex++) {
            out[cursor.offset(0)] = FastTranscendentals.fastExpF32(in[cursor.offset(1)]);
            if (outIndex + 1 < end) {
                cursor.step();
            }
        }
    }

    @Override
    protected void runIndexedArrayBF16(
            short[] in,
            float[] continuation,
            short[] out,
            UnaryStorageLayout layout,
            int start,
            int end
    ) {
        ElementwiseOffsetCursor cursor = layout.cursor(start);
        for (int outIndex = start; outIndex < end; outIndex++) {
            out[cursor.offset(0)] = TensorDTypeOps.toBFloat16Bits(
                    FastTranscendentals.fastExpF32(loadBF16(continuation, in, cursor.offset(1)))
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
            float[] out,
            UnaryStorageLayout layout,
            int start,
            int end
    ) {
        ElementwiseOffsetCursor cursor = layout.cursor(start);
        for (int outIndex = start; outIndex < end; outIndex++) {
            out[outIndex] = FastTranscendentals.fastExpF32(loadBF16(continuation, in, cursor.offset(1)));
            if (outIndex + 1 < end) {
                cursor.step();
            }
        }
    }

    @Override
    protected void runIndexedSegmentF64(
            MemorySegment in,
            MemorySegment out,
            UnaryStorageLayout layout,
            int start,
            int end
    ) {
        ElementwiseOffsetCursor cursor = layout.cursor(start);
        for (int outIndex = start; outIndex < end; outIndex++) {
            out.set(
                    JAVA_DOUBLE,
                    (long) cursor.offset(0) * Double.BYTES,
                    FastTranscendentals.fastExpF64(in.get(JAVA_DOUBLE, (long) cursor.offset(1) * Double.BYTES))
            );
            if (outIndex + 1 < end) {
                cursor.step();
            }
        }
    }

    @Override
    protected void runIndexedSegmentF32(
            MemorySegment in,
            MemorySegment out,
            UnaryStorageLayout layout,
            int start,
            int end
    ) {
        ElementwiseOffsetCursor cursor = layout.cursor(start);
        for (int outIndex = start; outIndex < end; outIndex++) {
            out.set(
                    JAVA_FLOAT,
                    (long) cursor.offset(0) * Float.BYTES,
                    FastTranscendentals.fastExpF32(in.get(JAVA_FLOAT, (long) cursor.offset(1) * Float.BYTES))
            );
            if (outIndex + 1 < end) {
                cursor.step();
            }
        }
    }

    @Override
    protected void runIndexedSegmentBF16(
            MemorySegment in,
            MemorySegment out,
            UnaryStorageLayout layout,
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
                    TensorDTypeOps.toBFloat16Bits(FastTranscendentals.fastExpF32(value))
            );
            if (outIndex + 1 < end) {
                cursor.step();
            }
        }
    }

    @Override
    protected void runIndexedMixedF64(
            CpuStorageView in,
            CpuStorageView out,
            UnaryStorageLayout layout,
            int start,
            int end
    ) {
        ElementwiseOffsetCursor cursor = layout.cursor(start);
        for (int outIndex = start; outIndex < end; outIndex++) {
            writeF64(out, cursor.offset(0), FastTranscendentals.fastExpF64(readF64(in, cursor.offset(1))));
            if (outIndex + 1 < end) {
                cursor.step();
            }
        }
    }

    @Override
    protected void runIndexedMixedF32(
            CpuStorageView in,
            CpuStorageView out,
            UnaryStorageLayout layout,
            int start,
            int end
    ) {
        ElementwiseOffsetCursor cursor = layout.cursor(start);
        for (int outIndex = start; outIndex < end; outIndex++) {
            writeF32(out, cursor.offset(0), FastTranscendentals.fastExpF32(readF32(in, cursor.offset(1))));
            if (outIndex + 1 < end) {
                cursor.step();
            }
        }
    }

    @Override
    protected void runIndexedMixedBF16(
            CpuStorageView in,
            CpuStorageView out,
            UnaryStorageLayout layout,
            int start,
            int end
    ) {
        ElementwiseOffsetCursor cursor = layout.cursor(start);
        for (int outIndex = start; outIndex < end; outIndex++) {
            writeBF16(out, cursor.offset(0), FastTranscendentals.fastExpF32(readBF16(in, cursor.offset(1))));
            if (outIndex + 1 < end) {
                cursor.step();
            }
        }
    }
}
