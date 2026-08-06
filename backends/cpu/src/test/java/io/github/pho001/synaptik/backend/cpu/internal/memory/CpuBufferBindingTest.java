package io.github.pho001.synaptik.backend.cpu.internal.memory;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.datatype.DataType;
import org.junit.jupiter.api.Test;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import java.lang.foreign.MemorySegment;

class CpuBufferBindingTest {
    @Test void classifiesNativeMemoryAsExactSegmentWithoutCopy() {
        try (var buffer = CpuNativeBuffer.allocate(DataType.FLOAT64, 16, 8)) {
            var argument = assertInstanceOf(CpuBufferArgument.Segment.class, buffer.argument());
            assertAll(
                    () -> assertSame(buffer.segment(), argument.segment()),
                    () -> assertEquals(16, argument.byteSize()),
                    () -> assertFalse(argument.readOnly()));
        }
    }

    @Test void classifiesEveryAdmittedHeapCarrierWithoutCopy() {
        assertAll(
                () -> assertInstanceOf(CpuBufferArgument.Doubles.class,
                        borrowed(DataType.FLOAT64, MemorySegment.ofArray(new double[2]), 2).argument()),
                () -> assertInstanceOf(CpuBufferArgument.Floats.class,
                        borrowed(DataType.FLOAT32, MemorySegment.ofArray(new float[2]), 2).argument()),
                () -> assertInstanceOf(CpuBufferArgument.Ints.class,
                        borrowed(DataType.INT32, MemorySegment.ofArray(new int[2]), 2).argument()),
                () -> assertInstanceOf(CpuBufferArgument.Longs.class,
                        borrowed(DataType.INT64, MemorySegment.ofArray(new long[2]), 2).argument()),
                () -> assertInstanceOf(CpuBufferArgument.Bytes.class,
                        borrowed(DataType.BOOL, MemorySegment.ofArray(new byte[2]), 2).argument()));
    }

    private static CpuBorrowedBuffer borrowed(DataType type, MemorySegment segment, long count) {
        return CpuBorrowedBuffer.borrow(new MemorySegmentStorage(type, count, segment));
    }
}
