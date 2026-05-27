package backend.cpu.kernels.layout;

import backend.cpu.storage.CpuStorageView;
import tensor.DataType;
import tensor.TensorMetadata;
import tensor.dtype.TensorDTypeOps;

import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;
import static java.lang.foreign.ValueLayout.JAVA_SHORT;

final class LayoutStorageSupport {
    private LayoutStorageSupport() {
    }

    static void validateInputViews(int expected, java.util.List<CpuStorageView> inputs, String opName) {
        if (inputs == null || inputs.size() != expected) {
            throw new IllegalArgumentException(opName + " expects exactly " + expected + " input storage view(s).");
        }
    }

    static void validateView(CpuStorageView view, DataType dtype, String label) {
        if (view == null) {
            throw new IllegalArgumentException(label + " storage view cannot be null.");
        }
        if (view.dtype() != dtype) {
            throw new IllegalStateException(label + " storage dtype mismatch. expected=" + dtype
                    + ", actual=" + view.dtype());
        }
    }

    static int[] denseStrides(int[] shape) {
        return TensorMetadata.computeStrides(shape);
    }

    static int offsetForLogical(int logical, int[] shape, int[] dense, int[] strides, int baseOffset) {
        int rem = logical;
        int offset = baseOffset;
        for (int d = 0; d < shape.length; d++) {
            int coord = rem / dense[d];
            rem %= dense[d];
            offset += coord * strides[d];
        }
        return offset;
    }

    static double readF64(CpuStorageView view, int offset) {
        if (view.isArray()) {
            return view.requireF64Array()[offset];
        }
        return view.requireSegment().get(JAVA_DOUBLE, (long) offset * Double.BYTES);
    }

    static void writeF64(CpuStorageView view, int offset, double value) {
        if (view.isArray()) {
            view.requireF64Array()[offset] = value;
            return;
        }
        view.requireSegment().set(JAVA_DOUBLE, (long) offset * Double.BYTES, value);
    }

    static float readF32(CpuStorageView view, int offset) {
        if (view.isArray()) {
            return view.requireF32Array()[offset];
        }
        return view.requireSegment().get(JAVA_FLOAT, (long) offset * Float.BYTES);
    }

    static void writeF32(CpuStorageView view, int offset, float value) {
        if (view.isArray()) {
            view.requireF32Array()[offset] = value;
            return;
        }
        view.requireSegment().set(JAVA_FLOAT, (long) offset * Float.BYTES, value);
    }

    static float readBF16AsF32(CpuStorageView view, int offset) {
        return TensorDTypeOps.fromBFloat16Bits(readBF16Bits(view, offset));
    }

    static short readBF16Bits(CpuStorageView view, int offset) {
        if (view.isArray()) {
            return view.requireBF16Array()[offset];
        }
        return view.requireSegment().get(JAVA_SHORT, (long) offset * Short.BYTES);
    }

    static void writeBF16(CpuStorageView view, int offset, float value) {
        writeBF16Bits(view, offset, TensorDTypeOps.toBFloat16Bits(value));
    }

    static void writeBF16Bits(CpuStorageView view, int offset, short bits) {
        if (view.isArray()) {
            view.requireBF16Array()[offset] = bits;
            return;
        }
        view.requireSegment().set(JAVA_SHORT, (long) offset * Short.BYTES, bits);
    }

    static int readI32(CpuStorageView view, int offset) {
        if (view.isArray()) {
            return view.requireI32Array()[offset];
        }
        return view.requireSegment().get(JAVA_INT, (long) offset * Integer.BYTES);
    }

    static void writeI32(CpuStorageView view, int offset, int value) {
        if (view.isArray()) {
            view.requireI32Array()[offset] = value;
            return;
        }
        view.requireSegment().set(JAVA_INT, (long) offset * Integer.BYTES, value);
    }

    static long readI64(CpuStorageView view, int offset) {
        if (view.isArray()) {
            return view.requireI64Array()[offset];
        }
        return view.requireSegment().get(JAVA_LONG, (long) offset * Long.BYTES);
    }

    static void writeI64(CpuStorageView view, int offset, long value) {
        if (view.isArray()) {
            view.requireI64Array()[offset] = value;
            return;
        }
        view.requireSegment().set(JAVA_LONG, (long) offset * Long.BYTES, value);
    }

    static byte readBool(CpuStorageView view, int offset) {
        byte value;
        if (view.isArray()) {
            value = view.requireBoolArray()[offset];
        } else {
            value = view.requireSegment().get(JAVA_BYTE, offset);
        }
        return value == 0 ? (byte) 0 : (byte) 1;
    }

    static void writeBool(CpuStorageView view, int offset, byte value) {
        byte normalized = value == 0 ? (byte) 0 : (byte) 1;
        if (view.isArray()) {
            view.requireBoolArray()[offset] = normalized;
            return;
        }
        view.requireSegment().set(JAVA_BYTE, offset, normalized);
    }

    static byte boolFromDouble(double value) {
        return value == 0.0d ? (byte) 0 : (byte) 1;
    }
}
