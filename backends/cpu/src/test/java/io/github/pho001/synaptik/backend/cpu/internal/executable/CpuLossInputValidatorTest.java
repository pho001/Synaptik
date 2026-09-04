package io.github.pho001.synaptik.backend.cpu.internal.executable;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.github.pho001.synaptik.backend.cpu.internal.lowering.CpuLossLowering;
import io.github.pho001.synaptik.backend.cpu.internal.memory.CpuBufferArgument;
import io.github.pho001.synaptik.model.datatype.DataType;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Test;

class CpuLossInputValidatorTest {
    @Test void scansStridedInt32TargetsAndSkipsIgnoredValuesBeforeBounds() {
        int ignored = -17;
        int[] target = {99, 1, ignored, 2, 99, 0, 1, 3, 99, 2, 0, 1};
        var geometry = geometry(new long[] {2, 4, 2}, 1, 1L, new long[] {5, 1}, true, ignored);

        assertDoesNotThrow(() -> CpuLossInputValidator.validate(
                new CpuBufferArgument.Ints(target, 0, 48, true), geometry));
    }

    @Test void rejectsNegativeAndOutOfRangeInt32AndInt64TargetsInArraysAndSegments() {
        var geometry = geometry(new long[] {1, 3}, 1, 0L, new long[] {1}, false, 0L);
        assertThrows(IndexOutOfBoundsException.class, () -> CpuLossInputValidator.validate(
                new CpuBufferArgument.Ints(new int[] {-1}, 0, 4, true), geometry));
        assertThrows(IndexOutOfBoundsException.class, () -> CpuLossInputValidator.validate(
                new CpuBufferArgument.Segment(DataType.INT32, MemorySegment.ofArray(new int[] {3}),
                        4, true), geometry));
        assertThrows(IndexOutOfBoundsException.class, () -> CpuLossInputValidator.validate(
                new CpuBufferArgument.Longs(new long[] {-1}, 0, 8, true), geometry));
        assertThrows(IndexOutOfBoundsException.class, () -> CpuLossInputValidator.validate(
                new CpuBufferArgument.Segment(DataType.INT64, MemorySegment.ofArray(new long[] {3}),
                        8, true), geometry));
    }

    @Test void admitsZeroClassesOnlyForEmptyOrAllIgnoredDomains() {
        var allIgnored = geometry(new long[] {2, 0, 2}, 1, 0L, new long[] {2, 1}, true, -1L);
        var empty = geometry(new long[] {0, 0, 2}, 1, 0L, new long[] {2, 1}, false, 0L);
        var nonIgnored = geometry(new long[] {1, 0, 1}, 1, 0L, new long[] {1, 1}, true, -1L);

        assertDoesNotThrow(() -> CpuLossInputValidator.validate(
                new CpuBufferArgument.Ints(new int[] {-1, -1, -1, -1}, 0, 16, true), allIgnored));
        assertDoesNotThrow(() -> CpuLossInputValidator.validate(
                new CpuBufferArgument.Longs(new long[] {99}, 0, 8, true), empty));
        assertThrows(IndexOutOfBoundsException.class, () -> CpuLossInputValidator.validate(
                new CpuBufferArgument.Segment(DataType.INT64, MemorySegment.ofArray(new long[] {0}),
                        8, true), nonIgnored));
    }

    private static CpuLossLowering.Geometry geometry(long[] extents, int axis, long targetOffset,
            long[] targetStrides, boolean ignorePresent, long ignoreValue) {
        return new CpuLossLowering.Geometry(extents, axis, extents.length - 1, 0,
                0L, targetOffset, 0L, contiguous(extents), targetStrides, new long[0],
                ignorePresent, ignoreValue);
    }

    private static long[] contiguous(long[] extents) {
        long[] strides = new long[extents.length];
        long stride = 1L;
        for (int index = extents.length - 1; index >= 0; index--) {
            strides[index] = stride;
            stride = Math.multiplyExact(stride, Math.max(1L, extents[index]));
        }
        return strides;
    }
}
