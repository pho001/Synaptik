package io.github.pho001.synaptik.backend.cpu.internal.executable;

import io.github.pho001.synaptik.backend.cpu.internal.ir.CpuAccessPlan;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBufferArgument;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Objects;

/**
 * Cold-binds canonical BOOL scans to direct byte storage and private traversal state.
 * Scans read current values before consumer writes, allocate nothing on success, and may be
 * repeated sequentially after either success or failure within the carrier's borrowed lifetime.
 */
public final class CpuBooleanInputValidator {
    private CpuBooleanInputValidator() { }

    /**
     * Captures one represented logical byte domain without reading its values.
     *
     * @param argument non-null already validated borrowed byte carrier
     * @param binding non-null selected logical range, including offsets and strides
     * @param message non-null diagnostic used when a byte is neither zero nor one
     * @return non-null sequential scan throwing IllegalArgumentException for noncanonical bytes
     * @throws NullPointerException if an argument is null
     * @throws IllegalArgumentException if the carrier is not byte-addressable BOOL storage
     */
    static Runnable bind(CpuBufferArgument argument, CpuAccessPlan.Binding binding, String message) {
        Objects.requireNonNull(binding, "binding");
        Objects.requireNonNull(message, "message");
        MemorySegment storage = storage(argument);
        long[] extents = binding.extents().stream().mapToLong(Long::longValue).toArray();
        long[] strides = binding.effectiveStrides().stream().mapToLong(Long::longValue).toArray();
        long[] coordinates = new long[extents.length];
        return () -> validateTemporal(storage, binding.startAddress(), extents, strides, coordinates,
                binding.start(), binding.end(), 0L, true, null, 0L, 0L, 0L, message);
    }

    /**
     * Captures the read addresses of an immutable affine-copy plan.
     *
     * @param argument non-null already validated borrowed byte carrier
     * @param pairs non-null immutable source/destination element-address pairs, retained directly
     * @param start inclusive logical pair bound
     * @param end exclusive logical pair bound within the prepared pair array
     * @return non-null allocation-free scan that rejects noncanonical source bytes
     * @throws NullPointerException if a required reference is null
     * @throws IllegalArgumentException if the carrier is not byte-addressable BOOL storage
     */
    static Runnable bindAffine(CpuBufferArgument argument, long[] pairs, long start, long end) {
        MemorySegment storage = storage(argument);
        Objects.requireNonNull(pairs, "pairs");
        return () -> validateAffineTemporal(storage, pairs, start, end, 0L,
                true, null, 0L, 0L, 0L);
    }

    /**
     * Checks a complete eligible logical range before its consuming unit writes any storage.
     * Addresses are bytes relative to the borrowed storage. No values are read for an absent
     * owner or inactive row; no wrappers, cursors or arrays are allocated. The caller must invoke
     * this once for the entire unit domain, not once immediately before each numerical range.
     *
     * @param storage borrowed byte storage, already bound and span-checked
     * @param startAddress address of logical {@code start}, before the temporal displacement
     * @param extents cold row-major logical extents, retained without mutation
     * @param strides cold effective byte strides, retained without mutation
     * @param coordinates invocation-private rank-sized cursor, reset on every call including retries
     * @param start inclusive logical ordinal
     * @param end exclusive logical ordinal
     * @param temporalOffset additional byte displacement for this step
     * @param ownerPresent whether this unit occurrence is structurally present, including empty domains
     * @param activity borrowed row-presence bytes, or null for all-active
     * @param activityOffset byte address of row zero in activity
     * @param rowStride logical ordinals per row coordinate; zero selects one shared presence byte
     * @param rowExtent extent of the proved original row axis, unused when rowStride is zero
     * @param message diagnostic for a noncanonical byte
     * @throws IllegalArgumentException if an eligible byte is not zero or one
     * @throws ArithmeticException if checked address arithmetic overflows
     */
    public static void validateTemporal(MemorySegment storage, long startAddress,
            long[] extents, long[] strides, long[] coordinates, long start, long end, long temporalOffset,
            boolean ownerPresent, MemorySegment activity, long activityOffset,
            long rowStride, long rowExtent, String message) {
        if (!ownerPresent) return;
        long address = Math.addExact(startAddress, temporalOffset);
        long remaining = start;
        for (int axis = extents.length - 1; axis >= 0; axis--) {
            coordinates[axis] = extents[axis] == 0 ? 0 : remaining % extents[axis];
            if (extents[axis] != 0) remaining /= extents[axis];
        }
        for (long logical = start; logical < end; logical++) {
            if (isActive(logical, activity, activityOffset, rowStride, rowExtent))
                check(storage.get(ValueLayout.JAVA_BYTE, address), message);
            for (int axis = coordinates.length - 1; axis >= 0; axis--) {
                coordinates[axis]++;
                address = Math.addExact(address, strides[axis]);
                if (coordinates[axis] < extents[axis]) break;
                coordinates[axis] = 0;
                address = Math.subtractExact(address, Math.multiplyExact(extents[axis], strides[axis]));
            }
        }
    }

    /**
     * Checks the source coordinates of a cold affine plan with temporal and row activity.
     *
     * @param storage borrowed already span-checked BOOL storage
     * @param pairs immutable interleaved source/destination byte addresses
     * @param start inclusive pair ordinal
     * @param end exclusive pair ordinal
     * @param temporalOffset source byte displacement for this step
     * @param ownerPresent whether the owning occurrence is structurally present
     * @param activity borrowed presence bytes, or null for all-active
     * @param activityOffset byte address of row zero
     * @param rowStride ordinals per row coordinate, or zero for a shared byte
     * @param rowExtent proved row extent, unused for a shared byte
     * @throws IllegalArgumentException if an eligible source byte is noncanonical
     * @throws ArithmeticException if checked address arithmetic overflows
     */
    public static void validateAffineTemporal(MemorySegment storage, long[] pairs,
            long start, long end, long temporalOffset, boolean ownerPresent,
            MemorySegment activity, long activityOffset, long rowStride, long rowExtent) {
        if (!ownerPresent) return;
        for (long logical = start; logical < end; logical++) {
            if (!isActive(logical, activity, activityOffset, rowStride, rowExtent)) continue;
            long address = Math.addExact(temporalOffset,
                    pairs[Math.toIntExact(Math.multiplyExact(logical, 2))]);
            check(storage.get(ValueLayout.JAVA_BYTE, address),
                    "BOOL condition must use canonical bytes");
        }
    }

    /**
     * Tests prebound row activity without reading operand storage. Whole-owner presence must be
     * checked separately: an empty row domain does not imply an absent all-valid occurrence.
     *
     * @param ordinal ordinal in the caller's represented logical domain
     * @param activity borrowed row bytes, or null for all-active
     * @param activityOffset address of row zero
     * @param rowStride ordinals per row coordinate, or zero for one shared presence byte
     * @param rowExtent positive proved row extent when rowStride is nonzero
     * @return whether the ordinal's row is present
     * @throws ArithmeticException if the presence address overflows
     */
    public static boolean isActive(long ordinal, MemorySegment activity, long activityOffset,
            long rowStride, long rowExtent) {
        return activity == null || activity.get(ValueLayout.JAVA_BYTE,
                Math.addExact(activityOffset,
                        rowStride == 0 ? 0 : ordinal / rowStride % rowExtent)) != 0;
    }

    private static MemorySegment storage(CpuBufferArgument argument) {
        return switch (Objects.requireNonNull(argument, "argument")) {
            case CpuBufferArgument.Bytes bytes -> MemorySegment.ofArray(bytes.carrier())
                    .asSlice(bytes.byteOffset(), bytes.byteSize());
            case CpuBufferArgument.Segment segment
                    when segment.dataType() == io.github.pho001.synaptik.model.datatype.DataType.BOOL
                    -> segment.segment();
            default -> throw new IllegalArgumentException("BOOL input requires a byte carrier");
        };
    }

    private static void check(byte value, String message) {
        if (value != 0 && value != 1) throw new IllegalArgumentException(message);
    }
}
