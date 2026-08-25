package io.github.pho001.synaptik.backend.cpu.internal.executable;

import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuSoftmaxLowering;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBufferArgument;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.foreign.ValueLayout;
import java.util.Objects;

/** Performs the complete CPU-private finite softmax-domain check before execution. */
public final class CpuSoftmaxInputValidator {
    private CpuSoftmaxInputValidator() { }

    /**
     * Validates every represented input and maximum shift without mutating storage.
     * @param input non-null already carrier-validated borrowed input
     * @param geometry non-null complete static softmax geometry
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalArgumentException if any input or type-specific maximum shift is non-finite
     * @throws ArithmeticException if checked address arithmetic overflows
     */
    public static void validate(CpuBufferArgument input, CpuSoftmaxLowering.Geometry geometry) {
        Objects.requireNonNull(input, "input"); Objects.requireNonNull(geometry, "geometry");
        if (geometry.elementCount() == 0) return;
        long[] extents = geometry.input().extents();
        long[] strides = geometry.input().strides();
        int axis = geometry.axis();
        for (long slice = 0; slice < geometry.sliceCount(); slice++) {
            long remaining = slice;
            long base = geometry.input().offset();
            for (int dimension = extents.length - 1; dimension >= 0; dimension--) {
                if (dimension == axis) continue;
                long coordinate = remaining % extents[dimension];
                remaining /= extents[dimension];
                base = Math.addExact(base, Math.multiplyExact(coordinate, strides[dimension]));
            }
            if (geometry.dataType() == DataType.BFLOAT16) validateBfloat(input, base,
                    strides[axis], geometry.sliceWidth());
            else validateWide(input, geometry.dataType(), base, strides[axis],
                    geometry.sliceWidth());
        }
    }

    private static void validateWide(CpuBufferArgument input, DataType type, long base,
            long stride, long width) {
        double maximum = Double.NEGATIVE_INFINITY;
        long address = base;
        for (long coordinate = 0; coordinate < width; coordinate++) {
            double value = readWide(input, type, address);
            if (!Double.isFinite(value)) throw new IllegalArgumentException(
                    "softmax input and shifts must be finite");
            if (value > maximum) maximum = value;
            address = Math.addExact(address, stride);
        }
        address = base;
        for (long coordinate = 0; coordinate < width; coordinate++) {
            if (!Double.isFinite(readWide(input, type, address) - maximum))
                throw new IllegalArgumentException("softmax input and shifts must be finite");
            address = Math.addExact(address, stride);
        }
    }

    private static void validateBfloat(CpuBufferArgument input, long base, long stride, long width) {
        float maximum = Float.NEGATIVE_INFINITY;
        long address = base;
        for (long coordinate = 0; coordinate < width; coordinate++) {
            float value = Float.intBitsToFloat(readShort(input, address) << 16);
            if (!Float.isFinite(value)) throw new IllegalArgumentException(
                    "softmax input and shifts must be finite");
            if (value > maximum) maximum = value;
            address = Math.addExact(address, stride);
        }
        address = base;
        for (long coordinate = 0; coordinate < width; coordinate++) {
            float value = Float.intBitsToFloat(readShort(input, address) << 16);
            if (!Float.isFinite(value - maximum)) throw new IllegalArgumentException(
                    "softmax input and shifts must be finite");
            address = Math.addExact(address, stride);
        }
    }

    private static double readWide(CpuBufferArgument input, DataType type, long address) {
        long base = input.byteOffset() / type.byteWidth();
        if (input instanceof CpuBufferArgument.Doubles value)
            return value.carrier()[Math.toIntExact(base + address)];
        if (input instanceof CpuBufferArgument.Floats value)
            return value.carrier()[Math.toIntExact(base + address)];
        var segment = ((CpuBufferArgument.Segment) input).segment();
        long bytes = Math.multiplyExact(address, type.byteWidth());
        return type == DataType.FLOAT64 ? segment.get(ValueLayout.JAVA_DOUBLE, bytes)
                : segment.get(ValueLayout.JAVA_FLOAT, bytes);
    }

    private static int readShort(CpuBufferArgument input, long address) {
        long base = input.byteOffset() / Short.BYTES;
        if (input instanceof CpuBufferArgument.Shorts value)
            return value.carrier()[Math.toIntExact(base + address)] & 0xffff;
        return ((CpuBufferArgument.Segment) input).segment().get(ValueLayout.JAVA_SHORT,
                Math.multiplyExact(address, Short.BYTES)) & 0xffff;
    }
}
