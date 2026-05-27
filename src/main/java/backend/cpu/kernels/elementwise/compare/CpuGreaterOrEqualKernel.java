package backend.cpu.kernels.elementwise.compare;

import backend.cpu.kernels.elementwise.ElementwiseOffsetCursor;
import backend.cpu.storage.CpuStorageView;
import operations.Operation;
import tensor.dtype.TensorDTypeOps;

import java.lang.foreign.MemorySegment;

import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

public final class CpuGreaterOrEqualKernel extends StorageAwareCompareElementwiseKernel {
    @Override
    protected Operation.OpType opType() {
        return Operation.OpType.GE;
    }

    @Override
    protected String opLabel() {
        return "ge";
    }

    @Override
    protected void runArrayF64(double[] left, double[] right, byte[] out, int start, int end) {
        for (int i = start; i < end; i++) {
            out[i] = bool(left[i] >= right[i]);
        }
    }

    @Override
    protected void runArrayF32(float[] left, float[] right, byte[] out, int start, int end) {
        for (int i = start; i < end; i++) {
            out[i] = bool(left[i] >= right[i]);
        }
    }

    @Override
    protected void runArrayBF16(short[] left, short[] right, byte[] out, int start, int end) {
        for (int i = start; i < end; i++) {
            float leftValue = TensorDTypeOps.fromBFloat16Bits(left[i]);
            float rightValue = TensorDTypeOps.fromBFloat16Bits(right[i]);
            out[i] = bool(leftValue >= rightValue);
        }
    }

    @Override
    protected void runSegmentF64(MemorySegment left, MemorySegment right, MemorySegment out, int start, int end) {
        for (int i = start; i < end; i++) {
            long offset = (long) i * Double.BYTES;
            out.set(JAVA_BYTE, i, bool(left.get(JAVA_DOUBLE, offset) >= right.get(JAVA_DOUBLE, offset)));
        }
    }

    @Override
    protected void runSegmentF32(MemorySegment left, MemorySegment right, MemorySegment out, int start, int end) {
        for (int i = start; i < end; i++) {
            long offset = (long) i * Float.BYTES;
            out.set(JAVA_BYTE, i, bool(left.get(JAVA_FLOAT, offset) >= right.get(JAVA_FLOAT, offset)));
        }
    }

    @Override
    protected void runSegmentBF16(MemorySegment left, MemorySegment right, MemorySegment out, int start, int end) {
        for (int i = start; i < end; i++) {
            long offset = (long) i * Short.BYTES;
            float leftValue = TensorDTypeOps.fromBFloat16Bits(left.get(JAVA_SHORT, offset));
            float rightValue = TensorDTypeOps.fromBFloat16Bits(right.get(JAVA_SHORT, offset));
            out.set(JAVA_BYTE, i, bool(leftValue >= rightValue));
        }
    }

    @Override
    protected void runIndexedArrayF64(
            double[] left,
            double[] right,
            byte[] out,
            CompareStorageLayout layout,
            int start,
            int end
    ) {
        ElementwiseOffsetCursor cursor = layout.cursor(start);
        for (int outIndex = start; outIndex < end; outIndex++) {
            out[cursor.offset(0)] = bool(left[cursor.offset(1)] >= right[cursor.offset(2)]);
            if (outIndex + 1 < end) {
                cursor.step();
            }
        }
    }

    @Override
    protected void runIndexedArrayF32(
            float[] left,
            float[] right,
            byte[] out,
            CompareStorageLayout layout,
            int start,
            int end
    ) {
        ElementwiseOffsetCursor cursor = layout.cursor(start);
        for (int outIndex = start; outIndex < end; outIndex++) {
            out[cursor.offset(0)] = bool(left[cursor.offset(1)] >= right[cursor.offset(2)]);
            if (outIndex + 1 < end) {
                cursor.step();
            }
        }
    }

    @Override
    protected void runIndexedArrayBF16(
            short[] left,
            short[] right,
            byte[] out,
            CompareStorageLayout layout,
            int start,
            int end
    ) {
        ElementwiseOffsetCursor cursor = layout.cursor(start);
        for (int outIndex = start; outIndex < end; outIndex++) {
            float leftValue = TensorDTypeOps.fromBFloat16Bits(left[cursor.offset(1)]);
            float rightValue = TensorDTypeOps.fromBFloat16Bits(right[cursor.offset(2)]);
            out[cursor.offset(0)] = bool(leftValue >= rightValue);
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
            CompareStorageLayout layout,
            int start,
            int end
    ) {
        ElementwiseOffsetCursor cursor = layout.cursor(start);
        for (int outIndex = start; outIndex < end; outIndex++) {
            double leftValue = left.get(JAVA_DOUBLE, (long) cursor.offset(1) * Double.BYTES);
            double rightValue = right.get(JAVA_DOUBLE, (long) cursor.offset(2) * Double.BYTES);
            out.set(JAVA_BYTE, cursor.offset(0), bool(leftValue >= rightValue));
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
            CompareStorageLayout layout,
            int start,
            int end
    ) {
        ElementwiseOffsetCursor cursor = layout.cursor(start);
        for (int outIndex = start; outIndex < end; outIndex++) {
            float leftValue = left.get(JAVA_FLOAT, (long) cursor.offset(1) * Float.BYTES);
            float rightValue = right.get(JAVA_FLOAT, (long) cursor.offset(2) * Float.BYTES);
            out.set(JAVA_BYTE, cursor.offset(0), bool(leftValue >= rightValue));
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
            CompareStorageLayout layout,
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
            out.set(JAVA_BYTE, cursor.offset(0), bool(leftValue >= rightValue));
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
            CompareStorageLayout layout,
            int start,
            int end
    ) {
        ElementwiseOffsetCursor cursor = layout.cursor(start);
        for (int outIndex = start; outIndex < end; outIndex++) {
            writeBool(out, cursor.offset(0), readF64(left, cursor.offset(1)) >= readF64(right, cursor.offset(2)));
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
            CompareStorageLayout layout,
            int start,
            int end
    ) {
        ElementwiseOffsetCursor cursor = layout.cursor(start);
        for (int outIndex = start; outIndex < end; outIndex++) {
            writeBool(out, cursor.offset(0), readF32(left, cursor.offset(1)) >= readF32(right, cursor.offset(2)));
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
            CompareStorageLayout layout,
            int start,
            int end
    ) {
        ElementwiseOffsetCursor cursor = layout.cursor(start);
        for (int outIndex = start; outIndex < end; outIndex++) {
            writeBool(out, cursor.offset(0), readBF16(left, cursor.offset(1)) >= readBF16(right, cursor.offset(2)));
            if (outIndex + 1 < end) {
                cursor.step();
            }
        }
    }
}
