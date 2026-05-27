package backend.cpu.kernels.nn;

import backend.cpu.storage.CpuStorageView;
import tensor.dtype.TensorDTypeOps;

import java.lang.foreign.MemorySegment;

import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

final class NormalizationStorageAccess {
    private NormalizationStorageAccess() {
    }

    static boolean allArrays(CpuStorageView first, CpuStorageView second, CpuStorageView third) {
        return first.isArray() && second.isArray() && third.isArray();
    }

    static boolean allArrays(CpuStorageView first, CpuStorageView second, CpuStorageView third, CpuStorageView fourth) {
        return first.isArray() && second.isArray() && third.isArray() && fourth.isArray();
    }

    static boolean allSegments(CpuStorageView first, CpuStorageView second, CpuStorageView third) {
        return first.isMemorySegment() && second.isMemorySegment() && third.isMemorySegment();
    }

    static boolean allSegments(CpuStorageView first, CpuStorageView second, CpuStorageView third, CpuStorageView fourth) {
        return first.isMemorySegment() && second.isMemorySegment() && third.isMemorySegment() && fourth.isMemorySegment();
    }

    static boolean isDenseContiguous(CpuStorageView view) {
        int[] shape = view.shape();
        int[] strides = view.strides();
        int expected = 1;
        for (int dim = shape.length - 1; dim >= 0; dim--) {
            if (strides[dim] != expected) {
                return false;
            }
            expected = Math.multiplyExact(expected, shape[dim]);
        }
        return true;
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

    static short readBF16Bits(short[] array, MemorySegment segment, int offset) {
        return array != null ? array[offset] : segment.get(JAVA_SHORT, (long) offset * Short.BYTES);
    }

    static float readBF16(short[] array, MemorySegment segment, int offset) {
        return TensorDTypeOps.fromBFloat16Bits(readBF16Bits(array, segment, offset));
    }

    static void writeBF16(short[] array, MemorySegment segment, int offset, float value) {
        short bits = TensorDTypeOps.toBFloat16Bits(value);
        if (array != null) {
            array[offset] = bits;
        } else {
            segment.set(JAVA_SHORT, (long) offset * Short.BYTES, bits);
        }
    }

    static float[] decodeBFloat16(CpuStorageView view, int base, int length) {
        short[] array = bf16Array(view);
        MemorySegment segment = bf16Segment(view);
        float[] decoded = new float[length];
        for (int i = 0; i < length; i++) {
            decoded[i] = readBF16(array, segment, base + i);
        }
        return decoded;
    }
}
