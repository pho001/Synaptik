package io.github.pho001.synaptik.backend.cpu.internal.memory;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.datatype.ScalarValue;
import io.github.pho001.synaptik.runtime.memory.BufferSlot;
import io.github.pho001.synaptik.runtime.memory.PreparedMemoryPlan;
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

    @Test void fp8ConstantCreatorsPreserveRawBitsAndIsolateEachRunAllocation() {
        var entry = new PreparedMemoryPlan.BufferEntry(new BufferSlot(0), 3, 64);
        for (var type : new DataType[] {DataType.FLOAT8_E4M3, DataType.FLOAT8_E5M2}) {
            for (byte bits : new byte[] {0, (byte) 0x80, 1, 0x7e, 0x7f, (byte) 0xff}) {
                var value = type == DataType.FLOAT8_E4M3
                        ? ScalarValue.float8E4m3Bits(bits) : ScalarValue.float8E5m2Bits(bits);
                var creator = CpuRepresentationRecipes.initializedBuffer(type, entry, value);
                try (var first = (CpuNativeBuffer) creator.create(3, 64);
                        var second = (CpuNativeBuffer) creator.create(3, 64)) {
                    assertEquals(type, first.dataType());
                    assertEquals(3, first.byteSize());
                    assertEquals(0, first.segment().address() % 64);
                    assertArrayEquals(new byte[] {bits, bits, bits},
                            first.segment().toArray(java.lang.foreign.ValueLayout.JAVA_BYTE));
                    first.segment().set(java.lang.foreign.ValueLayout.JAVA_BYTE, 0, (byte) 37);
                    assertArrayEquals(new byte[] {bits, bits, bits},
                            second.segment().toArray(java.lang.foreign.ValueLayout.JAVA_BYTE));
                }
                assertThrows(IllegalArgumentException.class, () -> creator.create(4, 64));
                assertThrows(IllegalArgumentException.class, () -> creator.create(3, 32));
            }
            var empty = new PreparedMemoryPlan.BufferEntry(new BufferSlot(0), 0, 64);
            var zero = type == DataType.FLOAT8_E4M3
                    ? ScalarValue.float8E4m3Bits((byte) 0x80)
                    : ScalarValue.float8E5m2Bits((byte) 0x80);
            try (var buffer = (CpuNativeBuffer) CpuRepresentationRecipes
                    .initializedBuffer(type, empty, zero).create(0, 64)) {
                assertEquals(0, buffer.byteSize());
            }
        }
    }
}
