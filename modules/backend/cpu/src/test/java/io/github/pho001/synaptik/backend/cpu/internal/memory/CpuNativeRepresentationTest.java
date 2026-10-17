package io.github.pho001.synaptik.backend.cpu.internal.memory;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Test;

class CpuNativeRepresentationTest {
    @Test void ownsExactAlignedNativeLifetimeIncludingZeroSize() {
        var buffer = CpuNativeBuffer.allocate(DataType.FLOAT64, 0, 64);
        assertAll(
                () -> assertEquals(0, buffer.byteSize()),
                () -> assertEquals(64, buffer.byteAlignment()),
                () -> assertEquals(0, buffer.segment().address() % 64));
        buffer.close();
        assertThrows(IllegalStateException.class, buffer::segment);
        assertDoesNotThrow(buffer::close);
    }


    @Test void segmentRouteRetainsExactHeapSliceWithoutHidingWrongTypedCarrier() {
        double[] values = {9, -0.0, Double.longBitsToDouble(0x7ff8000000000042L), 7};
        MemorySegment slice = MemorySegment.ofArray(values).asSlice(8, 16);
        var borrowed = CpuBorrowedBuffer.borrow(new MemorySegmentStorage(DataType.FLOAT64, 2,
                slice));
        assertSame(values, ((CpuBufferArgument.Doubles) borrowed.argument()).carrier());
        assertSame(slice, borrowed.segmentArgument().segment());
        assertEquals(0, borrowed.segmentArgument().byteOffset());
        assertEquals(16, borrowed.segmentArgument().byteSize());
        assertEquals(8, borrowed.segmentArgument().segment().address());

        var readOnly = CpuBorrowedBuffer.borrow(new MemorySegmentStorage(DataType.FLOAT64, 2,
                slice.asReadOnly()));
        assertTrue(readOnly.segmentArgument().readOnly());

        var wrong = CpuBorrowedBuffer.borrow(new MemorySegmentStorage(DataType.FLOAT64, 2,
                MemorySegment.ofArray(new long[2])));
        assertThrows(IllegalArgumentException.class, wrong::segmentArgument);
    }
}
