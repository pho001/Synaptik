package io.github.pho001.synaptik.backend.cpu.internal.executable;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuSoftmaxLowering;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBufferArgument;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Objects;

/**
 * Prebinds the CPU-private finite softmax-domain check without reading producer-owned values.
 * Each invocation scans complete slices on the orchestrator before consumer writes or submission.
 * FLOAT32 and FLOAT64 admit every finite represented input, including a maximum subtraction
 * that overflows to negative infinity. BFLOAT16 additionally requires finite working shifts,
 * while every finite FLOAT16 pair has a finite FLOAT32 shift. Geometry snapshots and borrowed
 * typed storage are retained only during binding; temporal calls create no wrappers or readers.
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
     *     BFLOAT16 shifts
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalArgumentException if the carrier or represented dtype is not floating
     */
    public static Runnable bind(CpuBufferArgument input, CpuSoftmaxLowering.Geometry geometry) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(geometry, "geometry");
        switch (geometry.dataType()) {
            case FLOAT64, FLOAT32, BFLOAT16, FLOAT16 -> { }
            default -> throw new IllegalArgumentException("softmax requires a floating dtype");
        }
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
        return () -> validateTemporal(storage, geometry.dataType(), geometry.input().offset(),
                extents, strides, geometry.axis(), geometry.sliceWidth(), 0L, geometry.sliceCount(),
                0L, true, null, 0L, 0L, 0L);
    }

    /**
     * Binds and immediately validates every represented input, plus the required BFLOAT16
     * maximum shifts. FLOAT32 and FLOAT64 admit working shifts that overflow to negative infinity,
     * and finite FLOAT16 inputs cannot overflow their FLOAT32 working shift.
     * Prepared execution uses {@link #bind} instead so producers run before the scan.
     *
     * @param input non-null already carrier-validated borrowed input
     * @param geometry non-null complete static softmax geometry
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalArgumentException if any input is non-finite or a BFLOAT16 maximum shift
     *     is non-finite
     * @throws ArithmeticException if checked address arithmetic overflows
     */
    public static void validate(CpuBufferArgument input, CpuSoftmaxLowering.Geometry geometry) {
        bind(input, geometry).run();
    }

    /**
     * Validates the entire eligible slice domain before any numerical range of a temporal unit.
     * This is the ordinary validator loop as well as a generated Scan call target. FLOAT32 and
     * FLOAT64 retain finite-input-only admission; only BFLOAT16 also checks working shifts.
     *
     * @param storage borrowed prebound typed storage; no wrapper is created
     * @param type cold floating represented dtype
     * @param base input layout offset in elements
     * @param extents immutable cold input extents
     * @param strides immutable cold input element strides
     * @param axis softmax axis
     * @param width complete slice width, never divided between validation calls
     * @param start inclusive slice ordinal
     * @param end exclusive slice ordinal for the complete unit validation domain
     * @param temporalOffset additional input element displacement for this step
     * @param ownerPresent structural occurrence presence, independent of empty domains
     * @param activity borrowed row bytes, or null for all-active
     * @param activityOffset byte address of row zero
     * @param rowStride slice ordinals per original row coordinate, or zero for a shared byte
     * @param rowExtent proved original row extent, unused for a shared byte
     * @throws IllegalArgumentException if an eligible input or required BFLOAT16 shift is nonfinite
     * @throws ArithmeticException if checked address arithmetic overflows
     */
    public static void validateTemporal(MemorySegment storage, DataType type, long base,
            long[] extents, long[] strides, int axis, long width, long start, long end,
            long temporalOffset, boolean ownerPresent, MemorySegment activity,
            long activityOffset, long rowStride, long rowExtent) {
        if (!ownerPresent) return;
        long origin = Math.addExact(base, temporalOffset);
        for (long slice = start; slice < end; slice++) {
            if (!CpuBooleanInputValidator.isActive(slice, activity, activityOffset,
                    rowStride, rowExtent)) continue;
            long remaining = slice;
            long address = origin;
            for (int dimension = extents.length - 1; dimension >= 0; dimension--) {
                if (dimension == axis) continue;
                long coordinate = remaining % extents[dimension];
                remaining /= extents[dimension];
                address = Math.addExact(address, Math.multiplyExact(coordinate, strides[dimension]));
            }
            switch (type) {
                case FLOAT64 -> validateDouble(storage, address, strides[axis], width);
                case FLOAT32 -> validateFloat(storage, address, strides[axis], width);
                case BFLOAT16 -> validateBfloat(storage, address, strides[axis], width);
                case FLOAT16 -> validateHalf(storage, address, strides[axis], width);
                default -> throw new IllegalArgumentException("softmax requires a floating dtype");
            }
        }
    }

    private static void validateDouble(MemorySegment input, long base, long stride, long width) {
        long address = base;
        for (long coordinate = 0; coordinate < width; coordinate++) {
            double value = input.get(ValueLayout.JAVA_DOUBLE, Math.multiplyExact(address, Double.BYTES));
            if (!Double.isFinite(value)) throw new IllegalArgumentException(
                    "softmax input must be finite");
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
