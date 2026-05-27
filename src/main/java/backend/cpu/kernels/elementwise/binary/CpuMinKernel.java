package backend.cpu.kernels.elementwise.binary;

import backend.cpu.kernels.elementwise.ElementwiseOffsetCursor;
import backend.cpu.kernels.elementwise.ElementwiseRangeLoop;
import backend.cpu.plan.elementwise.ResolvedDispatchHints;
import backend.cpu.storage.CpuStorageView;
import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import operations.Operation;
import tensor.dtype.TensorDTypeOps;

import java.lang.foreign.MemorySegment;

import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

public final class CpuMinKernel extends StorageAwareBinaryElementwiseKernel {
    private static final VectorSpecies<Double> F64 = DoubleVector.SPECIES_PREFERRED;
    private static final VectorSpecies<Float> F32 = FloatVector.SPECIES_PREFERRED;

    @Override
    protected Operation.OpType opType() {
        return Operation.OpType.MIN;
    }

    @Override
    protected String opLabel() {
        return "min";
    }

    @Override
    protected void runDirectF64(double[] left, double[] right, double[] out, ResolvedDispatchHints hints) {
        ElementwiseRangeLoop.run(out.length, hints, true,
                (start, end) -> runArrayF64(left, right, out, start, end),
                (start, end) -> runArrayVectorF64(left, right, out, start, end));
    }

    @Override
    protected void runDirectF32(float[] left, float[] right, float[] out, ResolvedDispatchHints hints) {
        ElementwiseRangeLoop.run(out.length, hints, true,
                (start, end) -> runArrayF32(left, right, out, start, end),
                (start, end) -> runArrayVectorF32(left, right, out, start, end));
    }

    @Override
    protected void runDirectBF16(
            short[] leftStorage,
            short[] rightStorage,
            float[] leftContinuation,
            float[] rightContinuation,
            short[] out,
            ResolvedDispatchHints hints
    ) {
        if (leftContinuation != null && rightContinuation != null) {
            ElementwiseRangeLoop.runScalar(out.length, hints,
                    (start, end) -> runArrayBF16(leftContinuation, rightContinuation, out, start, end));
        } else if (leftContinuation != null) {
            ElementwiseRangeLoop.runScalar(out.length, hints,
                    (start, end) -> runArrayBF16(leftContinuation, rightStorage, out, start, end));
        } else if (rightContinuation != null) {
            ElementwiseRangeLoop.runScalar(out.length, hints,
                    (start, end) -> runArrayBF16(leftStorage, rightContinuation, out, start, end));
        } else {
            ElementwiseRangeLoop.runScalar(out.length, hints,
                    (start, end) -> runArrayBF16(leftStorage, rightStorage, out, start, end));
        }
    }

    @Override
    protected void runDirectBF16ToFloat(
            short[] leftStorage,
            short[] rightStorage,
            float[] leftContinuation,
            float[] rightContinuation,
            float[] out,
            ResolvedDispatchHints hints
    ) {
        if (leftContinuation != null && rightContinuation != null) {
            runDirectF32(leftContinuation, rightContinuation, out, hints);
            return;
        }
        ElementwiseRangeLoop.runScalar(out.length, hints,
                (start, end) -> runArrayBF16ToFloat(
                        leftStorage,
                        rightStorage,
                        leftContinuation,
                        rightContinuation,
                        out,
                        start,
                        end
                ));
    }

    private static void runArrayF64(double[] left, double[] right, double[] out, int start, int end) {
        for (int i = start; i < end; i++) out[i] = Math.min(left[i], right[i]);
    }

    private static void runArrayVectorF64(double[] left, double[] right, double[] out, int start, int end) {
        int width = F64.length();
        int upper = end - ((end - start) % width);
        int i = start;
        for (; i < upper; i += width) {
            DoubleVector.fromArray(F64, left, i)
                    .lanewise(VectorOperators.MIN, DoubleVector.fromArray(F64, right, i))
                    .intoArray(out, i);
        }
        runArrayF64(left, right, out, i, end);
    }

    private static void runArrayF32(float[] left, float[] right, float[] out, int start, int end) {
        for (int i = start; i < end; i++) out[i] = Math.min(left[i], right[i]);
    }

    private static void runArrayVectorF32(float[] left, float[] right, float[] out, int start, int end) {
        int width = F32.length();
        int upper = end - ((end - start) % width);
        int i = start;
        for (; i < upper; i += width) {
            FloatVector.fromArray(F32, left, i)
                    .lanewise(VectorOperators.MIN, FloatVector.fromArray(F32, right, i))
                    .intoArray(out, i);
        }
        runArrayF32(left, right, out, i, end);
    }

    private static void runArrayBF16(float[] left, float[] right, short[] out, int start, int end) {
        for (int i = start; i < end; i++) out[i] = TensorDTypeOps.toBFloat16Bits(Math.min(left[i], right[i]));
    }

    private static void runArrayBF16(float[] left, short[] right, short[] out, int start, int end) {
        for (int i = start; i < end; i++) {
            out[i] = TensorDTypeOps.toBFloat16Bits(Math.min(left[i], TensorDTypeOps.fromBFloat16Bits(right[i])));
        }
    }

    private static void runArrayBF16(short[] left, float[] right, short[] out, int start, int end) {
        for (int i = start; i < end; i++) {
            out[i] = TensorDTypeOps.toBFloat16Bits(Math.min(TensorDTypeOps.fromBFloat16Bits(left[i]), right[i]));
        }
    }

    private static void runArrayBF16(short[] left, short[] right, short[] out, int start, int end) {
        for (int i = start; i < end; i++) {
            float leftValue = TensorDTypeOps.fromBFloat16Bits(left[i]);
            float rightValue = TensorDTypeOps.fromBFloat16Bits(right[i]);
            out[i] = TensorDTypeOps.toBFloat16Bits(Math.min(leftValue, rightValue));
        }
    }

    private static void runArrayBF16ToFloat(
            short[] leftStorage,
            short[] rightStorage,
            float[] leftContinuation,
            float[] rightContinuation,
            float[] out,
            int start,
            int end
    ) {
        for (int i = start; i < end; i++) {
            out[i] = Math.min(loadBF16(leftContinuation, leftStorage, i), loadBF16(rightContinuation, rightStorage, i));
        }
    }

    @Override
    protected void runSegmentF64(MemorySegment left, MemorySegment right, MemorySegment out, int start, int end) {
        for (int i = start; i < end; i++) {
            long offset = (long) i * Double.BYTES;
            out.set(JAVA_DOUBLE, offset, Math.min(left.get(JAVA_DOUBLE, offset), right.get(JAVA_DOUBLE, offset)));
        }
    }

    @Override
    protected void runSegmentF32(MemorySegment left, MemorySegment right, MemorySegment out, int start, int end) {
        for (int i = start; i < end; i++) {
            long offset = (long) i * Float.BYTES;
            out.set(JAVA_FLOAT, offset, Math.min(left.get(JAVA_FLOAT, offset), right.get(JAVA_FLOAT, offset)));
        }
    }

    @Override
    protected void runSegmentBF16(MemorySegment left, MemorySegment right, MemorySegment out, int start, int end) {
        for (int i = start; i < end; i++) {
            long offset = (long) i * Short.BYTES;
            float leftValue = TensorDTypeOps.fromBFloat16Bits(left.get(JAVA_SHORT, offset));
            float rightValue = TensorDTypeOps.fromBFloat16Bits(right.get(JAVA_SHORT, offset));
            out.set(JAVA_SHORT, offset, TensorDTypeOps.toBFloat16Bits(Math.min(leftValue, rightValue)));
        }
    }

    @Override
    protected void runIndexedArrayF64(
            double[] left,
            double[] right,
            double[] out,
            BinaryStorageLayout layout,
            int start,
            int end
    ) {
        ElementwiseOffsetCursor cursor = layout.cursor(start);
        for (int outIndex = start; outIndex < end; outIndex++) {
            out[cursor.offset(0)] = Math.min(left[cursor.offset(1)], right[cursor.offset(2)]);
            if (outIndex + 1 < end) {
                cursor.step();
            }
        }
    }

    @Override
    protected void runIndexedArrayF32(
            float[] left,
            float[] right,
            float[] out,
            BinaryStorageLayout layout,
            int start,
            int end
    ) {
        ElementwiseOffsetCursor cursor = layout.cursor(start);
        for (int outIndex = start; outIndex < end; outIndex++) {
            out[cursor.offset(0)] = Math.min(left[cursor.offset(1)], right[cursor.offset(2)]);
            if (outIndex + 1 < end) {
                cursor.step();
            }
        }
    }

    @Override
    protected void runIndexedArrayBF16(
            short[] left,
            short[] right,
            float[] leftContinuation,
            float[] rightContinuation,
            short[] out,
            BinaryStorageLayout layout,
            int start,
            int end
    ) {
        ElementwiseOffsetCursor cursor = layout.cursor(start);
        for (int outIndex = start; outIndex < end; outIndex++) {
            float leftValue = loadBF16(leftContinuation, left, cursor.offset(1));
            float rightValue = loadBF16(rightContinuation, right, cursor.offset(2));
            out[cursor.offset(0)] = TensorDTypeOps.toBFloat16Bits(Math.min(leftValue, rightValue));
            if (outIndex + 1 < end) {
                cursor.step();
            }
        }
    }

    @Override
    protected void runIndexedArrayBF16ToFloat(
            short[] left,
            short[] right,
            float[] leftContinuation,
            float[] rightContinuation,
            float[] out,
            BinaryStorageLayout layout,
            int start,
            int end
    ) {
        ElementwiseOffsetCursor cursor = layout.cursor(start);
        for (int outIndex = start; outIndex < end; outIndex++) {
            out[outIndex] = Math.min(
                    loadBF16(leftContinuation, left, cursor.offset(1)),
                    loadBF16(rightContinuation, right, cursor.offset(2))
            );
            if (outIndex + 1 < end) {
                cursor.step();
            }
        }
    }

    @Override
    protected void runIndexedSegmentF64(
            MemorySegment left,
            MemorySegment right,
            MemorySegment out,
            BinaryStorageLayout layout,
            int start,
            int end
    ) {
        ElementwiseOffsetCursor cursor = layout.cursor(start);
        for (int outIndex = start; outIndex < end; outIndex++) {
            long leftOffset = (long) cursor.offset(1) * Double.BYTES;
            long rightOffset = (long) cursor.offset(2) * Double.BYTES;
            long outOffset = (long) cursor.offset(0) * Double.BYTES;
            out.set(JAVA_DOUBLE, outOffset, Math.min(left.get(JAVA_DOUBLE, leftOffset), right.get(JAVA_DOUBLE, rightOffset)));
            if (outIndex + 1 < end) {
                cursor.step();
            }
        }
    }

    @Override
    protected void runIndexedSegmentF32(
            MemorySegment left,
            MemorySegment right,
            MemorySegment out,
            BinaryStorageLayout layout,
            int start,
            int end
    ) {
        ElementwiseOffsetCursor cursor = layout.cursor(start);
        for (int outIndex = start; outIndex < end; outIndex++) {
            long leftOffset = (long) cursor.offset(1) * Float.BYTES;
            long rightOffset = (long) cursor.offset(2) * Float.BYTES;
            long outOffset = (long) cursor.offset(0) * Float.BYTES;
            out.set(JAVA_FLOAT, outOffset, Math.min(left.get(JAVA_FLOAT, leftOffset), right.get(JAVA_FLOAT, rightOffset)));
            if (outIndex + 1 < end) {
                cursor.step();
            }
        }
    }

    @Override
    protected void runIndexedSegmentBF16(
            MemorySegment left,
            MemorySegment right,
            MemorySegment out,
            BinaryStorageLayout layout,
            int start,
            int end
    ) {
        ElementwiseOffsetCursor cursor = layout.cursor(start);
        for (int outIndex = start; outIndex < end; outIndex++) {
            float leftValue = TensorDTypeOps.fromBFloat16Bits(
                    left.get(JAVA_SHORT, (long) cursor.offset(1) * Short.BYTES)
            );
            float rightValue = TensorDTypeOps.fromBFloat16Bits(
                    right.get(JAVA_SHORT, (long) cursor.offset(2) * Short.BYTES)
            );
            out.set(
                    JAVA_SHORT,
                    (long) cursor.offset(0) * Short.BYTES,
                    TensorDTypeOps.toBFloat16Bits(Math.min(leftValue, rightValue))
            );
            if (outIndex + 1 < end) {
                cursor.step();
            }
        }
    }

    @Override
    protected void runIndexedMixedF64(
            CpuStorageView left,
            CpuStorageView right,
            CpuStorageView out,
            BinaryStorageLayout layout,
            int start,
            int end
    ) {
        ElementwiseOffsetCursor cursor = layout.cursor(start);
        for (int outIndex = start; outIndex < end; outIndex++) {
            writeF64(out, cursor.offset(0), Math.min(readF64(left, cursor.offset(1)), readF64(right, cursor.offset(2))));
            if (outIndex + 1 < end) {
                cursor.step();
            }
        }
    }

    @Override
    protected void runIndexedMixedF32(
            CpuStorageView left,
            CpuStorageView right,
            CpuStorageView out,
            BinaryStorageLayout layout,
            int start,
            int end
    ) {
        ElementwiseOffsetCursor cursor = layout.cursor(start);
        for (int outIndex = start; outIndex < end; outIndex++) {
            writeF32(out, cursor.offset(0), Math.min(readF32(left, cursor.offset(1)), readF32(right, cursor.offset(2))));
            if (outIndex + 1 < end) {
                cursor.step();
            }
        }
    }

    @Override
    protected void runIndexedMixedBF16(
            CpuStorageView left,
            CpuStorageView right,
            CpuStorageView out,
            BinaryStorageLayout layout,
            int start,
            int end
    ) {
        ElementwiseOffsetCursor cursor = layout.cursor(start);
        for (int outIndex = start; outIndex < end; outIndex++) {
            writeBF16(out, cursor.offset(0), Math.min(readBF16(left, cursor.offset(1)), readBF16(right, cursor.offset(2))));
            if (outIndex + 1 < end) {
                cursor.step();
            }
        }
    }
}
