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

    @Test void explicitStateAndCanonicalMaskUseDirectLongAndByteCarriers() {
        long[] state = {Long.MIN_VALUE, Long.MAX_VALUE}; byte[] mask = {0, 1};
        var stateArgument = assertInstanceOf(CpuBufferArgument.Longs.class,
                borrowed(DataType.INT64, MemorySegment.ofArray(state), 2).argument());
        var maskArgument = assertInstanceOf(CpuBufferArgument.Bytes.class,
                borrowed(DataType.BOOL, MemorySegment.ofArray(mask), 2).argument());
        assertAll(() -> assertSame(state, stateArgument.carrier()),
                () -> assertSame(mask, maskArgument.carrier()),
                () -> assertFalse(stateArgument.readOnly()),
                () -> assertFalse(maskArgument.readOnly()));
    }

    @Test void fp8HeapSlicesRetainRawBytesAndReadOnlyAccessWithoutBoolNormalization() {
        for (var type : new DataType[] {DataType.FLOAT8_E4M3, DataType.FLOAT8_E5M2}) {
            byte[] values = {11, (byte) 0x80, (byte) 0xff, 0x7f, 22};
            var slice = MemorySegment.ofArray(values).asSlice(1, 3).asReadOnly();
            var buffer = borrowed(type, slice, 3);
            var argument = buffer.argument();
            assertAll(() -> assertEquals(3, argument.byteSize()),
                    () -> assertTrue(argument.readOnly()),
                    () -> assertSame(slice, buffer.segmentArgument().segment()),
                    () -> assertEquals(type, buffer.segmentArgument().dataType()),
                    () -> assertArrayEquals(new byte[] {(byte) 0x80, (byte) 0xff, 0x7f},
                            buffer.segment().toArray(java.lang.foreign.ValueLayout.JAVA_BYTE)),
                    () -> assertThrows(IllegalArgumentException.class,
                            () -> buffer.segment().set(java.lang.foreign.ValueLayout.JAVA_BYTE, 0, (byte) 0)));
            values[2] = (byte) 0xfe;
            assertEquals((byte) 0xfe, buffer.segment().get(java.lang.foreign.ValueLayout.JAVA_BYTE, 1));
            buffer.close();
            assertEquals((byte) 0xfe, buffer.segment().get(java.lang.foreign.ValueLayout.JAVA_BYTE, 1));
        }
    }

    @Test void fp8RejectsObservableShortCarrierInsteadOfReinterpretingIt() {
        for (var type : new DataType[] {DataType.FLOAT8_E4M3, DataType.FLOAT8_E5M2}) {
            var buffer = borrowed(type, MemorySegment.ofArray(new short[1]), 2);
            assertThrows(IllegalArgumentException.class, buffer::argument);
            assertThrows(IllegalArgumentException.class, buffer::segmentArgument);
        }
    }

    private static CpuBorrowedBuffer borrowed(DataType type, MemorySegment segment, long count) {
        return CpuBorrowedBuffer.borrow(new MemorySegmentStorage(type, count, segment));
    }
}
