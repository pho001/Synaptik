package io.github.pho001.synaptik.backend.cpu.internal.executable;

import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuSoftmaxLowering;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBufferArgument;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Objects;
import java.util.function.LongConsumer;

/**
 * Prebinds the CPU-private finite softmax-domain check without reading producer-owned values.
 * Each invocation scans complete slices on the orchestrator before consumer writes or submission.
 * FLOAT32 admits every finite represented input, including a subtraction that overflows in
 * FLOAT32; FLOAT64 and BFLOAT16 additionally require finite working shifts, while every finite
 * FLOAT16 pair has a finite FLOAT32 shift. Geometry snapshots, carrier selection, and
 * slice-reader selection happen only during binding.
 */
public final class CpuSoftmaxInputValidator {
    private CpuSoftmaxInputValidator() { }

    /**
     * Binds an allocation-free value scan; binding itself does not read input values.
     *
     * @param input non-null already carrier-validated borrowed input, retained without copying
     * @param geometry non-null complete static softmax geometry
     * @return non-null scan of current values, valid only within the borrowed carrier lifetime;
     *     running it throws {@link IllegalArgumentException} for non-finite inputs or disallowed
     *     FLOAT64/BFLOAT16 shifts
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalArgumentException if the carrier or represented dtype is not floating
     */
    public static Runnable bind(CpuBufferArgument input, CpuSoftmaxLowering.Geometry geometry) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(geometry, "geometry");
        MemorySegment storage = switch (input) {
            case CpuBufferArgument.Doubles value -> MemorySegment.ofArray(value.carrier())
                    .asSlice(value.byteOffset(), value.byteSize());
            case CpuBufferArgument.Floats value -> MemorySegment.ofArray(value.carrier())
                    .asSlice(value.byteOffset(), value.byteSize());
            case CpuBufferArgument.Shorts value -> MemorySegment.ofArray(value.carrier())
                    .asSlice(value.byteOffset(), value.byteSize());
            case CpuBufferArgument.Segment value -> value.segment();
            default -> throw new IllegalArgumentException("softmax requires a floating carrier");
        };
        long[] extents = geometry.input().extents();
        long[] strides = geometry.input().strides();
        int axis = geometry.axis();
        long width = geometry.sliceWidth();
        long stride = strides[axis];
        LongConsumer sliceScan = switch (geometry.dataType()) {
            case FLOAT64 -> base -> validateDouble(storage, base, stride, width);
            case FLOAT32 -> base -> validateFloat(storage, base, stride, width);
            case BFLOAT16 -> base -> validateBfloat(storage, base, stride, width);
            case FLOAT16 -> base -> validateHalf(storage, base, stride, width);
            default -> throw new IllegalArgumentException("softmax requires a floating dtype");
        };
        return () -> {
            for (long slice = 0; slice < geometry.sliceCount(); slice++) {
                long remaining = slice;
                long base = geometry.input().offset();
                for (int dimension = extents.length - 1; dimension >= 0; dimension--) {
                    if (dimension == axis) continue;
                    long coordinate = remaining % extents[dimension];
                    remaining /= extents[dimension];
                    base = Math.addExact(base, Math.multiplyExact(coordinate, strides[dimension]));
                }
                sliceScan.accept(base);
            }
        };
    }

    /**
     * Binds and immediately validates every represented input, plus the required FLOAT64 and
     * BFLOAT16 maximum shifts. FLOAT32 admits a working shift that overflows to negative infinity,
     * and finite FLOAT16 inputs cannot overflow their FLOAT32 working shift.
     * Prepared execution uses {@link #bind} instead so producers run before the scan.
     *
     * @param input non-null already carrier-validated borrowed input
     * @param geometry non-null complete static softmax geometry
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalArgumentException if any input is non-finite or a FLOAT64/BFLOAT16 maximum
     *     shift is non-finite
     * @throws ArithmeticException if checked address arithmetic overflows
     */
    public static void validate(CpuBufferArgument input, CpuSoftmaxLowering.Geometry geometry) {
        bind(input, geometry).run();
    }

    private static void validateDouble(MemorySegment input, long base, long stride, long width) {
        double maximum = Double.NEGATIVE_INFINITY;
        long address = base;
        for (long coordinate = 0; coordinate < width; coordinate++) {
            double value = input.get(ValueLayout.JAVA_DOUBLE, Math.multiplyExact(address, Double.BYTES));
            if (!Double.isFinite(value)) throw new IllegalArgumentException(
                    "softmax input and shifts must be finite");
            if (value > maximum) maximum = value;
            address = Math.addExact(address, stride);
        }
        address = base;
        for (long coordinate = 0; coordinate < width; coordinate++) {
            double value = input.get(ValueLayout.JAVA_DOUBLE, Math.multiplyExact(address, Double.BYTES));
            if (!Double.isFinite(value - maximum)) throw new IllegalArgumentException(
                    "softmax input and shifts must be finite");
            address = Math.addExact(address, stride);
        }
    }

    private static void validateFloat(MemorySegment input, long base, long stride, long width) {
        long address = base;
        for (long coordinate = 0; coordinate < width; coordinate++) {
            float value = input.get(ValueLayout.JAVA_FLOAT, Math.multiplyExact(address, Float.BYTES));
            if (!Float.isFinite(value)) throw new IllegalArgumentException(
                    "softmax input must be finite");
            address = Math.addExact(address, stride);
        }
    }

    private static void validateBfloat(MemorySegment input, long base, long stride, long width) {
        float maximum = Float.NEGATIVE_INFINITY;
        long address = base;
        for (long coordinate = 0; coordinate < width; coordinate++) {
            float value = Float.intBitsToFloat(input.get(ValueLayout.JAVA_SHORT,
                    Math.multiplyExact(address, Short.BYTES)) << 16);
            if (!Float.isFinite(value)) throw new IllegalArgumentException(
                    "softmax input and shifts must be finite");
            if (value > maximum) maximum = value;
            address = Math.addExact(address, stride);
        }
        address = base;
        for (long coordinate = 0; coordinate < width; coordinate++) {
            float value = Float.intBitsToFloat(input.get(ValueLayout.JAVA_SHORT,
                    Math.multiplyExact(address, Short.BYTES)) << 16);
            if (!Float.isFinite(value - maximum)) throw new IllegalArgumentException(
                    "softmax input and shifts must be finite");
            address = Math.addExact(address, stride);
        }
    }

    private static void validateHalf(MemorySegment input, long base, long stride, long width) {
        long address = base;
        for (long coordinate = 0; coordinate < width; coordinate++) {
            float value = Float.float16ToFloat(input.get(ValueLayout.JAVA_SHORT,
                    Math.multiplyExact(address, Short.BYTES)));
            if (!Float.isFinite(value)) throw new IllegalArgumentException(
                    "softmax input must be finite");
            address = Math.addExact(address, stride);
        }
    }
}
