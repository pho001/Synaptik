package backend.cpu.kernels.elementwise.unary;

import backend.cpu.kernels.elementwise.ElementwiseOffsetCursor;
import backend.cpu.kernels.elementwise.ElementwiseRangeLoop;
import backend.cpu.kernels.elementwise.unary.support.CpuPowSupport;
import backend.cpu.plan.elementwise.ResolvedDispatchHints;
import backend.cpu.storage.CpuStorageView;
import operations.Operation;
import operations.elementwise.unary.pow;
import tensor.DataType;
import tensor.dtype.TensorDTypeOps;

import java.lang.foreign.MemorySegment;

import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

public final class CpuPowKernel extends StorageAwareScalarUnaryElementwiseKernel {
    @Override
    protected Operation.OpType opType() {
        return Operation.OpType.POW;
    }

    @Override
    protected String opLabel() {
        return "pow";
    }

    @Override
    protected ScalarUnaryParameters extractParameters(Operation operation) {
        pow power = (pow) operation;
        return new ScalarUnaryParameters(power.getExponent(), power.getExponentF32());
    }

    @Override
    protected boolean supportsNativeOpDType(DataType dtype) {
        return dtype == DataType.FLOAT32 || dtype == DataType.FLOAT64;
    }

    @Override
    protected void runDirectF64(double[] in, double parameter, double[] out, ResolvedDispatchHints hints) {
        ElementwiseRangeLoop.runScalar(out.length, hints,
                (start, end) -> runArrayF64(in, parameter, out, start, end));
    }

    @Override
    protected void runDirectF32(float[] in, float parameter, float[] out, ResolvedDispatchHints hints) {
        ElementwiseRangeLoop.runScalar(out.length, hints,
                (start, end) -> runArrayF32(in, parameter, out, start, end));
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
        for (int i = start; i < end; i++) out[i] = CpuPowSupport.applyF64(in[i], parameter);
    }

    private static void runArrayF32(float[] in, float parameter, float[] out, int start, int end) {
        for (int i = start; i < end; i++) out[i] = CpuPowSupport.applyF32(in[i], parameter);
    }

    private static void runArrayBF16(float[] in, float parameter, short[] out, int start, int end) {
        for (int i = start; i < end; i++) {
            out[i] = TensorDTypeOps.toBFloat16Bits(CpuPowSupport.applyF32(in[i], parameter));
        }
    }

    private static void runArrayBF16(short[] in, float parameter, short[] out, int start, int end) {
        for (int i = start; i < end; i++) {
            float value = TensorDTypeOps.fromBFloat16Bits(in[i]);
            out[i] = TensorDTypeOps.toBFloat16Bits(CpuPowSupport.applyF32(value, parameter));
        }
    }

    private static void runArrayBF16ToFloat(short[] in, float parameter, float[] out, int start, int end) {
        for (int i = start; i < end; i++) {
            out[i] = CpuPowSupport.applyF32(TensorDTypeOps.fromBFloat16Bits(in[i]), parameter);
        }
    }

    @Override
    protected void runSegmentF64(MemorySegment in, double parameter, MemorySegment out, int start, int end) {
        for (int i = start; i < end; i++) {
            long offset = (long) i * Double.BYTES;
            out.set(JAVA_DOUBLE, offset, CpuPowSupport.applyF64(in.get(JAVA_DOUBLE, offset), parameter));
        }
    }

    @Override
    protected void runSegmentF32(MemorySegment in, float parameter, MemorySegment out, int start, int end) {
        for (int i = start; i < end; i++) {
            long offset = (long) i * Float.BYTES;
            out.set(JAVA_FLOAT, offset, CpuPowSupport.applyF32(in.get(JAVA_FLOAT, offset), parameter));
        }
    }

    @Override
    protected void runSegmentBF16(MemorySegment in, float parameter, MemorySegment out, int start, int end) {
        for (int i = start; i < end; i++) {
            long offset = (long) i * Short.BYTES;
            float value = TensorDTypeOps.fromBFloat16Bits(in.get(JAVA_SHORT, offset));
            out.set(JAVA_SHORT, offset, TensorDTypeOps.toBFloat16Bits(CpuPowSupport.applyF32(value, parameter)));
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
            out[cursor.offset(0)] = CpuPowSupport.applyF64(in[cursor.offset(1)], parameter);
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
            out[cursor.offset(0)] = CpuPowSupport.applyF32(in[cursor.offset(1)], parameter);
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
                    CpuPowSupport.applyF32(loadBF16(continuation, in, cursor.offset(1)), parameter)
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
            out[outIndex] = CpuPowSupport.applyF32(loadBF16(continuation, in, cursor.offset(1)), parameter);
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
                    CpuPowSupport.applyF64(
                            in.get(JAVA_DOUBLE, (long) cursor.offset(1) * Double.BYTES),
                            parameter
                    )
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
                    CpuPowSupport.applyF32(
                            in.get(JAVA_FLOAT, (long) cursor.offset(1) * Float.BYTES),
                            parameter
                    )
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
                    TensorDTypeOps.toBFloat16Bits(CpuPowSupport.applyF32(value, parameter))
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
            writeF64(out, cursor.offset(0), CpuPowSupport.applyF64(readF64(in, cursor.offset(1)), parameter));
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
            writeF32(out, cursor.offset(0), CpuPowSupport.applyF32(readF32(in, cursor.offset(1)), parameter));
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
            writeBF16(out, cursor.offset(0), CpuPowSupport.applyF32(readBF16(in, cursor.offset(1)), parameter));
            if (outIndex + 1 < end) {
                cursor.step();
            }
        }
    }
}
