package io.github.pho001.synaptik.backend.cpu.internal.executable;

import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuLossLowering;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBufferArgument;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Objects;

/**
 * Cold-bound pre-write validator for index categorical loss targets.
 *
 * <p>The validator walks the complete non-class target domain in row-major logical order. It
 * compares the optional ignore value before checking a target's class bounds and never touches
 * logits or output storage. Binding captures typed storage and geometry without scanning values.
 * The orchestrator runs the scan after target producers and before any loss output write or
 * worker submission, including for a parallel {@code NONE} execution.</p>
 */
public final class CpuLossInputValidator {
    private CpuLossInputValidator() { }

    /**
     * Validates all non-ignored targets for one index categorical loss invocation.
     *
     * @param target non-null already carrier-validated INT32 or INT64 target argument
     * @param geometry non-null static loss geometry with an index target rank
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalArgumentException if the geometry is not index categorical, the carrier is
     *     not the represented integral target type
     * @throws IndexOutOfBoundsException if a non-ignored target is out of bounds
     * @throws ArithmeticException if checked coordinate or address arithmetic overflows
     */
    public static void validate(CpuBufferArgument target,
            CpuLossLowering.Geometry geometry) {
        bind(target, geometry).run();
    }

    /**
     * Captures an allocation-free target scan without reading producer-owned values.
     *
     * @param target non-null already carrier-validated borrowed INT32 or INT64 storage
     * @param geometry non-null complete static index categorical geometry
     * @return non-null action that checks current targets before each consumer invocation;
     *     invalid non-ignored targets cause {@link IndexOutOfBoundsException}
     * @throws NullPointerException if an argument is {@code null}
     * @throws IllegalArgumentException if the geometry or carrier is not index categorical
     * @throws ArithmeticException if checked sample-count arithmetic overflows
     */
    public static Runnable bind(CpuBufferArgument target, CpuLossLowering.Geometry geometry) {
        Objects.requireNonNull(target, "target");
        Objects.requireNonNull(geometry, "geometry");
        long[] extents = geometry.extents();
        int axis = geometry.axis();
        if (axis < 0 || geometry.targetRank() != extents.length - 1) {
            throw new IllegalArgumentException("index categorical loss geometry required");
        }
        DataType targetType = targetType(target);
        long[] strides = geometry.targetStrides();
        long samples = sampleCount(extents, axis);
        MemorySegment storage = switch (target) {
            case CpuBufferArgument.Ints value -> MemorySegment.ofArray(value.carrier())
                    .asSlice(value.byteOffset(), value.byteSize());
            case CpuBufferArgument.Longs value -> MemorySegment.ofArray(value.carrier())
                    .asSlice(value.byteOffset(), value.byteSize());
            case CpuBufferArgument.Segment value -> value.segment();
            default -> throw new IllegalArgumentException("index loss requires integral storage");
        };
        return () -> validateTemporal(storage, targetType.byteWidth(), geometry.targetOffset(),
                extents, strides, axis, geometry.ignorePresent(), geometry.ignoreValue(),
                0L, samples, 0L, true, null, 0L, 0L, 0L);
    }

    /**
     * Checks the entire eligible non-class target domain before a temporal loss unit writes.
     * Ignore comparison precedes bounds, including for zero classes. No logits are read and no
     * finite-value policy is added. Ordinary bound validation uses this same loop.
     *
     * @param storage borrowed prebound integral storage
     * @param targetBytes cold represented target width, either four or eight bytes
     * @param base target layout element offset
     * @param extents immutable cold logits extents
     * @param strides immutable cold target element strides
     * @param axis logits class axis, excluded from sample ordinals
     * @param ignorePresent whether the ignore value is enabled
     * @param ignoreValue ignored target, compared before bounds
     * @param start inclusive sample ordinal
     * @param end exclusive sample ordinal for the entire unit domain
     * @param temporalOffset additional target element displacement for this step
     * @param ownerPresent structural occurrence presence, independent of empty domains
     * @param activity borrowed row bytes, or null for all-active
     * @param activityOffset byte address of row zero
     * @param rowStride sample ordinals per original row coordinate, or zero for a shared byte
     * @param rowExtent proved row extent, unused for a shared byte
     * @throws IndexOutOfBoundsException if an eligible non-ignored target is outside class bounds
     * @throws ArithmeticException if checked address arithmetic overflows
     */
    public static void validateTemporal(MemorySegment storage, int targetBytes, long base,
            long[] extents, long[] strides, int axis, boolean ignorePresent, long ignoreValue,
            long start, long end, long temporalOffset, boolean ownerPresent,
            MemorySegment activity, long activityOffset, long rowStride, long rowExtent) {
        if (!ownerPresent) return;
        long origin = Math.addExact(base, temporalOffset);
        for (long ordinal = start; ordinal < end; ordinal++) {
            if (!CpuBooleanInputValidator.isActive(ordinal, activity, activityOffset,
                    rowStride, rowExtent)) continue;
            long address = address(ordinal, origin, extents, strides, axis);
            long value = targetBytes == Integer.BYTES
                    ? storage.get(ValueLayout.JAVA_INT, Math.multiplyExact(address, Integer.BYTES))
                    : storage.get(ValueLayout.JAVA_LONG, Math.multiplyExact(address, Long.BYTES));
            check(value, ordinal, extents[axis], ignorePresent, ignoreValue);
        }
    }

    private static long address(long ordinal, long base, long[] extents, long[] strides, int axis) {
        long address = base;
        for (int dimension = extents.length - 1; dimension >= 0; dimension--) {
            if (dimension == axis) continue;
            long coordinate = ordinal % extents[dimension];
            ordinal /= extents[dimension];
            int targetDimension = dimension < axis ? dimension : dimension - 1;
            address = Math.addExact(address, Math.multiplyExact(coordinate, strides[targetDimension]));
        }
        return address;
    }

    private static void check(long value, long ordinal, long classes,
            boolean ignorePresent, long ignoreValue) {
        if (ignorePresent && value == ignoreValue) return;
        if (value < 0 || value >= classes) {
            throw new IndexOutOfBoundsException("index categorical target at logical position "
                    + ordinal + " is out of bounds: value=" + value + ", classes=" + classes);
        }
    }

    private static long sampleCount(long[] extents, int axis) {
        long result = 1L;
        for (int dimension = 0; dimension < extents.length; dimension++) {
            if (dimension == axis) continue;
            if (extents[dimension] == 0L) return 0L;
            result = Math.multiplyExact(result, extents[dimension]);
        }
        return result;
    }

    private static DataType targetType(CpuBufferArgument target) {
        if (target instanceof CpuBufferArgument.Ints) return DataType.INT32;
        if (target instanceof CpuBufferArgument.Longs) return DataType.INT64;
        if (target instanceof CpuBufferArgument.Segment segment
                && (segment.dataType() == DataType.INT32 || segment.dataType() == DataType.INT64)) {
            return segment.dataType();
        }
        throw new IllegalArgumentException("index categorical target requires an INT32 or INT64 carrier");
    }

}
