package io.github.pho001.synaptik.backend.cpu.internal.executable;

import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuLossLowering;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBufferArgument;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.foreign.ValueLayout;
import java.util.Objects;

/**
 * Cold pre-write validator for index categorical loss targets.
 *
 * <p>The validator walks the complete non-class target domain in row-major logical order. It
 * compares the optional ignore value before checking a target's class bounds and never touches
 * logits or output storage. Calling it during prepared-executable binding therefore makes an
 * invalid target an all-or-nothing invocation failure, including for a parallel {@code NONE}
 * execution.</p>
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
     *     not the represented integral target type, or a non-ignored target is out of bounds
     * @throws ArithmeticException if checked coordinate or address arithmetic overflows
     */
    public static void validate(CpuBufferArgument target,
            CpuLossLowering.Geometry geometry) {
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
        long classes = extents[axis];
        long base = Math.addExact(target.byteOffset() / targetType.byteWidth(),
                geometry.targetOffset());
        for (long ordinal = 0; ordinal < samples; ordinal++) {
            long remaining = ordinal;
            long address = base;
            for (int dimension = extents.length - 1; dimension >= 0; dimension--) {
                if (dimension == axis) continue;
                long extent = extents[dimension];
                long coordinate = remaining % extent;
                remaining /= extent;
                int targetDimension = dimension < axis ? dimension : dimension - 1;
                address = Math.addExact(address,
                        Math.multiplyExact(coordinate, strides[targetDimension]));
            }
            long value = read(target, targetType, address);
            if (geometry.ignorePresent() && value == geometry.ignoreValue()) continue;
            if (value < 0 || value >= classes) {
                throw new IndexOutOfBoundsException("index categorical target at logical position "
                        + ordinal + " is out of bounds: value=" + value + ", classes=" + classes);
            }
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

    private static long read(CpuBufferArgument target, DataType type, long address) {
        if (type == DataType.INT32) {
            if (target instanceof CpuBufferArgument.Ints ints)
                return ints.carrier()[Math.toIntExact(address)];
            return ((CpuBufferArgument.Segment) target).segment().get(ValueLayout.JAVA_INT,
                    Math.multiplyExact(address, Integer.BYTES));
        }
        if (target instanceof CpuBufferArgument.Longs longs) return longs.carrier()[Math.toIntExact(address)];
        return ((CpuBufferArgument.Segment) target).segment().get(ValueLayout.JAVA_LONG,
                Math.multiplyExact(address, Long.BYTES));
    }
}
