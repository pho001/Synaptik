package io.github.pho001.synaptik.backend.cpu.internal.memory;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.DynamicDimension;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Optional;
import java.util.concurrent.Executors;
import org.junit.jupiter.api.Test;

final class CpuHostSnapshotExporterTest {
    @Test
    void exportsAllHeapCarriersWithExactCanonicalBits() {
        long doubleBits = 0xfff8_0000_0000_0042L;
        int floatBits = 0xffc0_0042;
        assertArrayEquals(bytes(8).putLong(doubleBits).array(), copy(DataType.FLOAT64,
                new double[] {Double.longBitsToDouble(doubleBits)}, 1));
        assertArrayEquals(bytes(4).putInt(floatBits).array(), copy(DataType.FLOAT32,
                new float[] {Float.intBitsToFloat(floatBits)}, 1));
        assertArrayEquals(bytes(6).putShort((short) 0x7fc1).putShort((short) 0x8000)
                        .putShort((short) 0x3f80).array(),
                copy(DataType.BFLOAT16, new short[] {(short) 0x7fc1, (short) 0x8000,
                        (short) 0x3f80}, 3));
        assertArrayEquals(bytes(6).putShort((short) 0x7e01).putShort((short) 0x8000)
                        .putShort((short) 0x3c00).array(),
                copy(DataType.FLOAT16, new short[] {(short) 0x7e01, (short) 0x8000,
                        (short) 0x3c00}, 3));
        assertArrayEquals(bytes(16).putLong(Long.MIN_VALUE).putLong(Long.MAX_VALUE).array(),
                copy(DataType.INT64, new long[] {Long.MIN_VALUE, Long.MAX_VALUE}, 2));
        assertArrayEquals(bytes(8).putInt(Integer.MIN_VALUE).putInt(Integer.MAX_VALUE).array(),
                copy(DataType.INT32, new int[] {Integer.MIN_VALUE, Integer.MAX_VALUE}, 2));
        assertArrayEquals(new byte[] {0, 1}, copy(DataType.BOOL, new byte[] {0, 1}, 2));
    }

    @Test
    void exportsEveryFp8EncodingExactlyFromHeapNativeAndStridedSlices() {
        byte[] values = new byte[256];
        for (int index = 0; index < values.length; index++) values[index] = (byte) index;
        for (var type : new DataType[] {DataType.FLOAT8_E4M3, DataType.FLOAT8_E5M2}) {
            assertArrayEquals(values, copy(type, values, values.length));
            try (var nativeBuffer = CpuNativeBuffer.allocate(type, values.length, 64)) {
                MemorySegment.copy(MemorySegment.ofArray(values), 0,
                        nativeBuffer.segment(), 0, values.length);
                var shape = Shape.of(256);
                assertArrayEquals(values, CpuHostSnapshotExporter.copy(nativeBuffer,
                        descriptor(type, shape, LayoutDescriptor.contiguous(shape)), 256));
            }
            var slice = MemorySegment.ofArray(values).asSlice(1, 254).asReadOnly();
            var borrowed = CpuBorrowedBuffer.borrow(new MemorySegmentStorage(type, 254, slice));
            var shape = Shape.of(2, 2);
            var view = descriptor(type, shape,
                    LayoutDescriptor.of(shape, new long[] {0, 2}, 125, true));
            assertArrayEquals(new byte[] {126, (byte) 128, 126, (byte) 128},
                    CpuHostSnapshotExporter.copy(borrowed, view, 4));
        }
    }

    @Test
    void exportsEveryTypeFromNativeOrderSegmentStorage() {
        assertNative(DataType.FLOAT64, 8, segment -> segment.set(
                ValueLayout.JAVA_DOUBLE_UNALIGNED.withOrder(ByteOrder.nativeOrder()), 0,
                Double.longBitsToDouble(0x8000_0000_0000_0000L)),
                bytes(8).putLong(0x8000_0000_0000_0000L).array());
        assertNative(DataType.FLOAT32, 4, segment -> segment.set(
                ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.nativeOrder()), 0,
                Float.intBitsToFloat(0x7fc0_0123)), bytes(4).putInt(0x7fc0_0123).array());
        assertNative(DataType.BFLOAT16, 2, segment -> segment.set(
                ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.nativeOrder()), 0,
                (short) 0xff80), bytes(2).putShort((short) 0xff80).array());
        assertNative(DataType.FLOAT16, 2, segment -> segment.set(
                ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.nativeOrder()), 0,
                (short) 0xfc01), bytes(2).putShort((short) 0xfc01).array());
        assertNative(DataType.INT64, 8, segment -> segment.set(
                ValueLayout.JAVA_LONG_UNALIGNED.withOrder(ByteOrder.nativeOrder()), 0,
                0x0102_0304_0506_0708L), bytes(8).putLong(0x0102_0304_0506_0708L).array());
        assertNative(DataType.INT32, 4, segment -> segment.set(
                ValueLayout.JAVA_INT_UNALIGNED.withOrder(ByteOrder.nativeOrder()), 0,
                0x0102_0304), bytes(4).putInt(0x0102_0304).array());
        assertNative(DataType.BOOL, 1,
                segment -> segment.set(ValueLayout.JAVA_BYTE, 0, (byte) 1), new byte[] {1});
    }

    @Test
    void followsOffsetStridesBroadcastScalarAndEmptyGeometry() {
        int[] values = {10, 11, 12, 13, 14, 15, 16, 17};
        CpuBorrowedBuffer buffer = borrowed(DataType.INT32, values, values.length);
        Shape stridedShape = Shape.of(2, 2);
        TensorDescriptor strided = new TensorDescriptor(DataType.INT32, stridedShape,
                Optional.of(LayoutDescriptor.of(stridedShape, new long[] {3, 2}, 1, true)), false);
        assertArrayEquals(bytes(16).putInt(11).putInt(13).putInt(14).putInt(16).array(),
                CpuHostSnapshotExporter.copy(buffer, strided, 16));

        Shape broadcastShape = Shape.of(2, 3);
        TensorDescriptor broadcast = new TensorDescriptor(DataType.INT32, broadcastShape,
                Optional.of(LayoutDescriptor.of(broadcastShape, new long[] {0, 1}, 2, true)), false);
        assertArrayEquals(bytes(24).putInt(12).putInt(13).putInt(14)
                        .putInt(12).putInt(13).putInt(14).array(),
                CpuHostSnapshotExporter.copy(buffer, broadcast, 24));

        TensorDescriptor scalar = new TensorDescriptor(DataType.INT32, Shape.scalar(),
                Optional.of(LayoutDescriptor.of(Shape.scalar(), new long[0], 5, true)), false);
        assertArrayEquals(bytes(4).putInt(15).array(),
                CpuHostSnapshotExporter.copy(buffer, scalar, 4));

        TensorDescriptor empty = new TensorDescriptor(DataType.INT32, Shape.of(0, Long.MAX_VALUE),
                Optional.of(LayoutDescriptor.of(Shape.of(0, Long.MAX_VALUE),
                        new long[] {Long.MAX_VALUE, Long.MAX_VALUE}, Long.MAX_VALUE, true)), false);
        CpuNativeBuffer zero = CpuNativeBuffer.allocate(DataType.INT32, 0, 4);
        try (zero) {
            byte[] first = CpuHostSnapshotExporter.copy(zero, empty, 0);
            byte[] second = CpuHostSnapshotExporter.copy(zero, empty, 0);
            assertEquals(0, first.length);
            assertNotSame(first, second);
        }
    }

    @Test
    void denseAndGeneralTraversalProduceIdenticalBytes() {
        long[] values = {1, 2, 3, 4};
        CpuBorrowedBuffer buffer = borrowed(DataType.INT64, values, values.length);
        Shape shape = Shape.of(2, 2);
        TensorDescriptor dense = descriptor(DataType.INT64, shape, LayoutDescriptor.contiguous(shape));
        TensorDescriptor general = descriptor(DataType.INT64, shape,
                LayoutDescriptor.of(shape, new long[] {2, 1}, 0, true));
        assertArrayEquals(CpuHostSnapshotExporter.copy(buffer, dense, 32),
                CpuHostSnapshotExporter.copy(buffer, general, 32));
    }

    @Test
    void denseExportRetainsSelectedHeapRegionAndUnalignedNativeRawBits() {
        float[] values = {1.0f, Float.intBitsToFloat(0xffc0_0123),
                Float.intBitsToFloat(0x8000_0000), 2.0f};
        MemorySegment selected = MemorySegment.ofArray(values).asSlice(4, 8);
        Shape shape = Shape.of(2);
        TensorDescriptor float32 = descriptor(DataType.FLOAT32, shape, LayoutDescriptor.contiguous(shape));
        byte[] expected = bytes(8).putInt(0xffc0_0123).putInt(0x8000_0000).array();
        // Mutable sliced arrays expose a carrier-relative offset; read-only heap segments do not.
        for (MemorySegment region : new MemorySegment[] {selected, selected.asReadOnly()}) {
            var borrowed = CpuBorrowedBuffer.borrow(new MemorySegmentStorage(DataType.FLOAT32, 2, region));
            assertArrayEquals(expected, CpuHostSnapshotExporter.copy(borrowed, float32, 8));
        }
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment unaligned = arena.allocate(17, 1).asSlice(1, 16);
            unaligned.set(ValueLayout.JAVA_LONG_UNALIGNED, 0, 0x7ff8_0000_0000_0042L);
            unaligned.set(ValueLayout.JAVA_LONG_UNALIGNED, 8, Long.MIN_VALUE);
            var borrowed = CpuBorrowedBuffer.borrow(new MemorySegmentStorage(DataType.FLOAT64, 2, unaligned));
            TensorDescriptor float64 = descriptor(DataType.FLOAT64, shape, LayoutDescriptor.contiguous(shape));
            assertArrayEquals(bytes(16).putLong(0x7ff8_0000_0000_0042L).putLong(Long.MIN_VALUE).array(),
                    CpuHostSnapshotExporter.copy(borrowed, float64, 16));
        }
    }

    @Test
    void validatesLogicalBoolBytesWithoutReadingInactivePoison() {
        CpuBorrowedBuffer buffer = borrowed(DataType.BOOL, new byte[] {2, 1, (byte) 255}, 3);
        Shape shape = Shape.of(3);
        TensorDescriptor valid = descriptor(DataType.BOOL, shape,
                LayoutDescriptor.of(shape, new long[] {0}, 1, true));
        assertArrayEquals(new byte[] {1, 1, 1}, CpuHostSnapshotExporter.copy(buffer, valid, 3));
        TensorDescriptor invalid = descriptor(DataType.BOOL, shape,
                LayoutDescriptor.of(shape, new long[] {0}, 2, true));
        assertThrows(IllegalArgumentException.class,
                () -> CpuHostSnapshotExporter.copy(buffer, invalid, 3));
    }

    @Test
    void rejectsInvalidMaterializationGeometryBeforeAccessingClosedStorage() {
        CpuNativeBuffer closed = CpuNativeBuffer.allocate(DataType.BOOL, 1, 1);
        closed.close();
        TensorDescriptor scalar = descriptor(DataType.BOOL, Shape.scalar(),
                LayoutDescriptor.contiguous(Shape.scalar()));
        // A prematurely entered data-copy path would instead report closed storage.
        assertThrows(IllegalArgumentException.class, () -> CpuHostSnapshotExporter.copy(closed, scalar, -1));
        assertThrows(IllegalArgumentException.class, () -> CpuHostSnapshotExporter.copy(closed, scalar, 0));
        Shape dynamic = Shape.ofDimensions(new DynamicDimension("N"));
        TensorDescriptor dynamicDescriptor = new TensorDescriptor(DataType.BOOL, dynamic, Optional.empty(), false);
        assertThrows(IllegalArgumentException.class,
                () -> CpuHostSnapshotExporter.copy(closed, dynamicDescriptor, 0));
        TensorDescriptor unresolved = new TensorDescriptor(DataType.BOOL, Shape.of(1), Optional.empty(), false);
        assertThrows(IllegalArgumentException.class, () -> CpuHostSnapshotExporter.copy(closed, unresolved, 1));
        Shape tooLarge = Shape.of((long) Integer.MAX_VALUE + 1);
        TensorDescriptor ceiling = descriptor(DataType.BOOL, tooLarge, LayoutDescriptor.contiguous(tooLarge));
        assertThrows(IllegalArgumentException.class, () -> CpuHostSnapshotExporter.copy(closed, ceiling, Long.MAX_VALUE));
    }

    @Test
    void rejectsLogicalCountAndByteCountOverflow() {
        Shape countOverflow = Shape.of(Long.MAX_VALUE, 2);
        TensorDescriptor resolvedCount = descriptor(DataType.BOOL, countOverflow,
                LayoutDescriptor.of(countOverflow, new long[] {0, 0}, 0, true));
        assertThrows(ArithmeticException.class,
                () -> CpuHostSnapshotExporter.copy(() -> { }, resolvedCount, Long.MAX_VALUE));

        Shape byteOverflow = Shape.of(Long.MAX_VALUE / 8 + 1);
        TensorDescriptor descriptor = descriptor(DataType.FLOAT64, byteOverflow,
                LayoutDescriptor.contiguous(byteOverflow));
        assertThrows(ArithmeticException.class,
                () -> CpuHostSnapshotExporter.copy(() -> { }, descriptor, Long.MAX_VALUE));
    }

    @Test
    void rejectsUnsupportedClosedInaccessibleOrIncompatibleStorage() throws Exception {
        Shape one = Shape.of(1);
        TensorDescriptor int32 = descriptor(DataType.INT32, one, LayoutDescriptor.contiguous(one));
        TestCpuBuffer subclass = new TestCpuBuffer(DataType.INT32, 4,
                MemorySegment.ofArray(new int[] {1}));
        assertThrows(IllegalArgumentException.class, () -> CpuHostSnapshotExporter.copy(subclass, int32, 4));

        CpuNativeBuffer closed = CpuNativeBuffer.allocate(DataType.INT32, 4, 4);
        closed.close();
        assertThrows(IllegalStateException.class, () -> CpuHostSnapshotExporter.copy(closed, int32, 4));

        try (Arena arena = Arena.ofConfined(); var executor = Executors.newSingleThreadExecutor()) {
            CpuBorrowedBuffer confined = CpuBorrowedBuffer.borrow(new MemorySegmentStorage(
                    DataType.INT32, 1, arena.allocate(4, 4)));
            Throwable cause = assertThrows(java.util.concurrent.ExecutionException.class,
                    () -> executor.submit(() ->
                            CpuHostSnapshotExporter.copy(confined, int32, 4)).get()).getCause();
            assertInstanceOf(IllegalStateException.class, cause);
        }

        CpuBorrowedBuffer wrongType = borrowed(DataType.FLOAT32, new float[] {1}, 1);
        assertThrows(IllegalArgumentException.class, () -> CpuHostSnapshotExporter.copy(wrongType, int32, 4));

        try (CpuNativeBuffer partial = CpuNativeBuffer.allocate(DataType.INT32, 3, 1)) {
            assertThrows(IllegalArgumentException.class, () -> CpuHostSnapshotExporter.copy(partial, int32, 4));
        }
        try (CpuNativeBuffer empty = CpuNativeBuffer.allocate(DataType.INT32, 0, 4)) {
            assertThrows(IllegalArgumentException.class, () -> CpuHostSnapshotExporter.copy(empty, int32, 4));
        }
        CpuBorrowedBuffer wrongCarrier = CpuBorrowedBuffer.borrow(new MemorySegmentStorage(
                DataType.FLOAT32, 1, MemorySegment.ofArray(new int[] {1})));
        TensorDescriptor float32 = descriptor(DataType.FLOAT32, one,
                LayoutDescriptor.contiguous(one));
        assertThrows(IllegalArgumentException.class, () -> CpuHostSnapshotExporter.copy(wrongCarrier, float32, 4));
    }

    @Test
    void equalCapacityAndLimitSucceedAndExtraCapacityIsIgnored() {
        CpuBorrowedBuffer exact = borrowed(DataType.INT32, new int[] {7}, 1);
        TensorDescriptor one = descriptor(DataType.INT32, Shape.of(1),
                LayoutDescriptor.contiguous(Shape.of(1)));
        assertArrayEquals(bytes(4).putInt(7).array(),
                CpuHostSnapshotExporter.copy(exact, one, 4));
        CpuBorrowedBuffer extra = borrowed(DataType.INT32, new int[] {7, 8}, 2);
        assertArrayEquals(bytes(4).putInt(7).array(),
                CpuHostSnapshotExporter.copy(extra, one, 4));
    }

    private static byte[] copy(DataType type, Object array, int count) {
        return CpuHostSnapshotExporter.copy(borrowed(type, array, count),
                descriptor(type, Shape.of(count), LayoutDescriptor.contiguous(Shape.of(count))),
                (long) count * type.byteWidth());
    }

    private static CpuBorrowedBuffer borrowed(DataType type, Object unused, int count) {
        MemorySegment segment = switch (unused) {
            case double[] values -> MemorySegment.ofArray(values);
            case float[] values -> MemorySegment.ofArray(values);
            case short[] values -> MemorySegment.ofArray(values);
            case int[] values -> MemorySegment.ofArray(values);
            case long[] values -> MemorySegment.ofArray(values);
            case byte[] values -> MemorySegment.ofArray(values);
            default -> throw new AssertionError(unused);
        };
        return CpuBorrowedBuffer.borrow(new MemorySegmentStorage(type, count, segment));
    }

    private static TensorDescriptor descriptor(DataType type, Shape shape, LayoutDescriptor layout) {
        return new TensorDescriptor(type, shape, Optional.of(layout), false);
    }

    private static ByteBuffer bytes(int size) {
        return ByteBuffer.allocate(size).order(ByteOrder.BIG_ENDIAN);
    }

    private static void assertNative(DataType type, int width,
            java.util.function.Consumer<MemorySegment> initializer, byte[] expected) {
        try (CpuNativeBuffer buffer = CpuNativeBuffer.allocate(type, width, width)) {
            initializer.accept(buffer.segment());
            assertArrayEquals(expected, CpuHostSnapshotExporter.copy(buffer,
                    descriptor(type, Shape.of(1), LayoutDescriptor.contiguous(Shape.of(1))), width));
        }
    }


    private static final class TestCpuBuffer extends CpuBufferRepresentation {
        private TestCpuBuffer(DataType type, long size, MemorySegment segment) {
            super(type, size, segment);
        }

        @Override protected boolean isClosed() { return false; }

        @Override public void close() { }
    }
}
