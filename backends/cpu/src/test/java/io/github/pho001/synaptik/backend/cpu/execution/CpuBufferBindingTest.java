package io.github.pho001.synaptik.backend.cpu.execution;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import java.lang.foreign.MemorySegment;
import java.lang.reflect.Modifier;
import java.util.List;
import org.junit.jupiter.api.Test;

class CpuBufferBindingTest {
    @Test void executionFoundationHasNoPublicTopLevelType() {
        for (Class<?> type : List.of(CpuBufferArgument.class, CpuBufferRepresentation.class,
                CpuBorrowedBuffer.class, CpuNativeBuffer.class, CpuNativeWorkspace.class,
                CpuPreparedExecutable.class, CpuWorkerGroup.class, CpuRangeBody.class,
                CpuParallelExecutionException.class)) {
            assertFalse(Modifier.isPublic(type.getModifiers()), type.getName());
        }
    }

    @Test void classifiesAllSixExactHeapCarriersWithoutCopying() {
        var carriers = List.of(new double[2], new float[2], new short[2], new int[2], new long[2], new byte[2]);
        var expected = List.of(CpuBufferArgument.Doubles.class, CpuBufferArgument.Floats.class,
                CpuBufferArgument.Shorts.class, CpuBufferArgument.Ints.class,
                CpuBufferArgument.Longs.class, CpuBufferArgument.Bytes.class);
        DataType[] types = DataType.values();
        for (int index = 0; index < types.length; index++) {
            MemorySegment segment = ofArray(carriers.get(index));
            var storage = new MemorySegmentStorage(types[index], 2, segment);
            var borrowed = CpuBorrowedBuffer.borrow(storage);
            CpuBufferArgument argument = borrowed.argument();
            Class<?> expectedType = expected.get(index);
            assertAll(
                    () -> assertSame(storage, borrowed.storage()),
                    () -> assertSame(segment, borrowed.segment()),
                    () -> assertEquals(expectedType, argument.getClass()),
                    () -> assertEquals(segment.byteSize(), argument.byteSize()),
                    () -> assertEquals(segment.address(), argument.byteOffset()));
            borrowed.close();
            assertFalse(borrowed.isClosed());
        }
    }

    @Test void preservesWritableHeapSliceOffsetAndOpaqueSegmentIdentity() {
        float[] values = new float[8];
        MemorySegment slice = MemorySegment.ofArray(values).asSlice(8, 16);
        var heap = CpuBorrowedBuffer.borrow(new MemorySegmentStorage(DataType.FLOAT32, 4, slice));
        var floats = assertInstanceOf(CpuBufferArgument.Floats.class, heap.argument());
        MemorySegment readOnlyHeap = MemorySegment.ofArray(new float[4]).asReadOnly();
        var opaqueHeap = CpuBorrowedBuffer.borrow(
                new MemorySegmentStorage(DataType.FLOAT32, 4, readOnlyHeap));
        var opaqueArgument = assertInstanceOf(CpuBufferArgument.Segment.class,
                opaqueHeap.argument());
        var nativeBuffer = CpuNativeBuffer.allocate(DataType.INT32, 16, 16);
        try {
            var nativeArgument = assertInstanceOf(CpuBufferArgument.Segment.class,
                    nativeBuffer.argument());
            assertAll(
                    () -> assertSame(values, floats.carrier()),
                    () -> assertEquals(8, floats.byteOffset()),
                    () -> assertFalse(floats.readOnly()),
                    () -> assertSame(readOnlyHeap, opaqueArgument.segment()),
                    () -> assertTrue(opaqueArgument.readOnly()),
                    () -> assertEquals(0, opaqueArgument.byteOffset()),
                    () -> assertSame(nativeBuffer.segment(), nativeArgument.segment()),
                    () -> assertEquals(0, nativeArgument.byteOffset()));
        } finally { nativeBuffer.close(); }
    }

    @Test void rejectsWrongHeapCarrierForLogicalType() {
        var storage = new MemorySegmentStorage(DataType.INT32, 1, MemorySegment.ofArray(new float[1]));
        var borrowed = assertDoesNotThrow(() -> CpuBorrowedBuffer.borrow(storage));
        assertThrows(IllegalArgumentException.class, borrowed::argument);
    }

    private static MemorySegment ofArray(Object carrier) {
        if (carrier instanceof double[] v) return MemorySegment.ofArray(v);
        if (carrier instanceof float[] v) return MemorySegment.ofArray(v);
        if (carrier instanceof short[] v) return MemorySegment.ofArray(v);
        if (carrier instanceof int[] v) return MemorySegment.ofArray(v);
        if (carrier instanceof long[] v) return MemorySegment.ofArray(v);
        return MemorySegment.ofArray((byte[]) carrier);
    }
}
