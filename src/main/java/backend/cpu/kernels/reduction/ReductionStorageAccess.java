package backend.cpu.kernels.reduction;

import backend.cpu.storage.CpuStorageView;

import java.lang.foreign.MemorySegment;

import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

final class ReductionStorageAccess {
    private ReductionStorageAccess() {
    }

    static int logicalToOffset(int logicalIndex, int[] shape, int[] strides, int storageOffset) {
        int remaining = logicalIndex;
        int offset = storageOffset;
        for (int dim = shape.length - 1; dim >= 0; dim--) {
            int coord = remaining % shape[dim];
            remaining /= shape[dim];
            offset += coord * strides[dim];
        }
        return offset;
    }

    static double[] f64Array(CpuStorageView view) {
        return view.isArray() ? view.requireF64Array() : null;
    }

    static MemorySegment f64Segment(CpuStorageView view) {
        return view.isMemorySegment() ? view.requireSegment() : null;
    }

    static double readF64(double[] array, MemorySegment segment, int offset) {
        return array != null ? array[offset] : segment.get(JAVA_DOUBLE, (long) offset * Double.BYTES);
    }

    static void writeF64(double[] array, MemorySegment segment, int offset, double value) {
        if (array != null) {
            array[offset] = value;
        } else {
            segment.set(JAVA_DOUBLE, (long) offset * Double.BYTES, value);
        }
    }

    static float[] f32Array(CpuStorageView view) {
        return view.isArray() ? view.requireF32Array() : null;
    }

    static MemorySegment f32Segment(CpuStorageView view) {
        return view.isMemorySegment() ? view.requireSegment() : null;
    }

    static float readF32(float[] array, MemorySegment segment, int offset) {
        return array != null ? array[offset] : segment.get(JAVA_FLOAT, (long) offset * Float.BYTES);
    }

    static void writeF32(float[] array, MemorySegment segment, int offset, float value) {
        if (array != null) {
            array[offset] = value;
        } else {
            segment.set(JAVA_FLOAT, (long) offset * Float.BYTES, value);
        }
    }

    static short[] bf16Array(CpuStorageView view) {
        return view.isArray() ? view.requireBF16Array() : null;
    }

    static MemorySegment bf16Segment(CpuStorageView view) {
        return view.isMemorySegment() ? view.requireSegment() : null;
    }

    static short readBF16(short[] array, MemorySegment segment, int offset) {
        return array != null ? array[offset] : segment.get(JAVA_SHORT, (long) offset * Short.BYTES);
    }

    static void writeBF16(short[] array, MemorySegment segment, int offset, short value) {
        if (array != null) {
            array[offset] = value;
        } else {
            segment.set(JAVA_SHORT, (long) offset * Short.BYTES, value);
        }
    }

    static int[] i32Array(CpuStorageView view) {
        return view.isArray() ? view.requireI32Array() : null;
    }

    static MemorySegment i32Segment(CpuStorageView view) {
        return view.isMemorySegment() ? view.requireSegment() : null;
    }

    static int readI32(int[] array, MemorySegment segment, int offset) {
        return array != null ? array[offset] : segment.get(JAVA_INT, (long) offset * Integer.BYTES);
    }

    static void writeI32(int[] array, MemorySegment segment, int offset, int value) {
        if (array != null) {
            array[offset] = value;
        } else {
            segment.set(JAVA_INT, (long) offset * Integer.BYTES, value);
        }
    }

    static long[] i64Array(CpuStorageView view) {
        return view.isArray() ? view.requireI64Array() : null;
    }

    static MemorySegment i64Segment(CpuStorageView view) {
        return view.isMemorySegment() ? view.requireSegment() : null;
    }

    static long readI64(long[] array, MemorySegment segment, int offset) {
        return array != null ? array[offset] : segment.get(JAVA_LONG, (long) offset * Long.BYTES);
    }

    static void writeI64(long[] array, MemorySegment segment, int offset, long value) {
        if (array != null) {
            array[offset] = value;
        } else {
            segment.set(JAVA_LONG, (long) offset * Long.BYTES, value);
        }
    }
}
