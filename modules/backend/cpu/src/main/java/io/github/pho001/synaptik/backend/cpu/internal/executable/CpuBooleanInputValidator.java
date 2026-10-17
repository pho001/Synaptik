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
final class CpuBooleanInputValidator {
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
        long[] initial = binding.startCoordinates().stream().mapToLong(Long::longValue).toArray();
        long[] coordinates = new long[initial.length];
        return () -> {
            System.arraycopy(initial, 0, coordinates, 0, initial.length);
            long address = binding.startAddress();
            for (long logical = binding.start(); logical < binding.end(); logical++) {
                check(storage.get(ValueLayout.JAVA_BYTE, address), message);
                for (int axis = coordinates.length - 1; axis >= 0; axis--) {
                    coordinates[axis]++;
                    address = Math.addExact(address, strides[axis]);
                    if (coordinates[axis] < extents[axis]) break;
                    coordinates[axis] = 0;
                    address = Math.subtractExact(address,
                            Math.multiplyExact(extents[axis], strides[axis]));
                }
            }
        };
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
        return () -> {
            for (long logical = start; logical < end; logical++) {
                long address = pairs[Math.toIntExact(Math.multiplyExact(logical, 2))];
                check(storage.get(ValueLayout.JAVA_BYTE, address),
                        "BOOL condition must use canonical bytes");
            }
        };
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
