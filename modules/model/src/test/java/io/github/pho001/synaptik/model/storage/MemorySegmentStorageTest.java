package io.github.pho001.synaptik.model.storage;

import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static org.junit.jupiter.api.Assertions.assertAll;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.model.datatype.DataType;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Map;
import org.junit.jupiter.api.Test;

class MemorySegmentStorageTest {
    @Test
    void calculatesExactByteSizeForAllDataTypesAndRetainsInputs() {
        Map<DataType, MemorySegment> segments = Map.of(
                DataType.FLOAT64, MemorySegment.ofArray(new double[3]),
                DataType.FLOAT32, MemorySegment.ofArray(new float[3]),
                DataType.BFLOAT16, MemorySegment.ofArray(new short[3]),
                DataType.INT32, MemorySegment.ofArray(new int[3]),
                DataType.INT64, MemorySegment.ofArray(new long[3]),
                DataType.BOOL, MemorySegment.ofArray(new byte[3]));

        for (DataType dataType : DataType.values()) {
            MemorySegment segment = segments.get(dataType);
            MemorySegmentStorage storage = new MemorySegmentStorage(dataType, 3, segment);

            assertAll(
                    () -> assertSame(dataType, storage.dataType()),
                    () -> assertEquals(3, storage.elementCapacity()),
                    () -> assertEquals(3L * dataType.byteWidth(), storage.byteSize()),
                    () -> assertEquals(segment.byteSize(), storage.byteSize()),
                    () -> assertSame(segment, storage.segment()),
                    () -> assertFalse(storage.isReadOnly()),
                    () -> assertTrue(storage.isAlive()));
        }
    }

    @Test
    void validatesInTheSpecifiedOrderWithExactMessages() {
        MemorySegment oneByte = MemorySegment.ofArray(new byte[1]);

        NullPointerException nullDataType = assertThrows(
                NullPointerException.class,
                () -> new MemorySegmentStorage(null, -1, null));
        NullPointerException nullSegment = assertThrows(
                NullPointerException.class,
                () -> new MemorySegmentStorage(DataType.FLOAT64, -1, null));
        IllegalArgumentException negativeCapacity = assertThrows(
                IllegalArgumentException.class,
                () -> new MemorySegmentStorage(DataType.FLOAT64, -1, oneByte));
        ArithmeticException overflow = assertThrows(
                ArithmeticException.class,
                () -> new MemorySegmentStorage(DataType.FLOAT64, Long.MAX_VALUE, oneByte));
        IllegalArgumentException wrongSize = assertThrows(
                IllegalArgumentException.class,
                () -> new MemorySegmentStorage(DataType.FLOAT32, 1, oneByte));

        assertAll(
                () -> assertEquals("dataType", nullDataType.getMessage()),
                () -> assertEquals("segment", nullSegment.getMessage()),
                () -> assertEquals(
                        "elementCapacity must be non-negative: -1",
                        negativeCapacity.getMessage()),
                () -> assertEquals(
                        "element byte size overflows long: elementCapacity=9223372036854775807, byteWidth=8",
                        overflow.getMessage()),
                () -> assertEquals(
                        "segment byte size must equal required byte size: required=4, actual=1",
                        wrongSize.getMessage()));
    }

    @Test
    void acceptsZeroCapacityAndArithmeticMaximumBoundaries() {
        MemorySegmentStorage zero =
                new MemorySegmentStorage(DataType.BOOL, 0, MemorySegment.NULL);
        long float64Maximum = Long.MAX_VALUE / DataType.FLOAT64.byteWidth();
        MemorySegment tinySegment = MemorySegment.ofArray(new byte[1]);

        assertAll(
                () -> assertEquals(0, zero.elementCapacity()),
                () -> assertEquals(0, zero.byteSize()),
                () -> assertSame(MemorySegment.NULL, zero.segment()),
                () -> assertTrue(zero.isAlive()),
                () -> {
                    IllegalArgumentException maximumDoesNotOverflow = assertThrows(
                            IllegalArgumentException.class,
                            () -> new MemorySegmentStorage(
                                    DataType.FLOAT64, float64Maximum, tinySegment));
                    assertEquals(
                            "segment byte size must equal required byte size: required=9223372036854775800, actual=1",
                            maximumDoesNotOverflow.getMessage());
                },
                () -> {
                    ArithmeticException aboveMaximum = assertThrows(
                            ArithmeticException.class,
                            () -> new MemorySegmentStorage(
                                    DataType.FLOAT64, float64Maximum + 1, tinySegment));
                    assertEquals(
                            "element byte size overflows long: elementCapacity=1152921504606846976, byteWidth=8",
                            aboveMaximum.getMessage());
                },
                () -> {
                    IllegalArgumentException boolMaximumDoesNotOverflow = assertThrows(
                            IllegalArgumentException.class,
                            () -> new MemorySegmentStorage(
                                    DataType.BOOL, Long.MAX_VALUE, tinySegment));
                    assertEquals(
                            "segment byte size must equal required byte size: required=9223372036854775807, actual=1",
                            boolMaximumDoesNotOverflow.getMessage());
                });
    }

    @Test
    void rejectsZeroPositiveUndersizedOversizedAndNonDivisibleMismatches() {
        assertAll(
                () -> assertThrows(
                        IllegalArgumentException.class,
                        () -> new MemorySegmentStorage(
                                DataType.FLOAT32, 1, MemorySegment.NULL)),
                () -> assertThrows(
                        IllegalArgumentException.class,
                        () -> new MemorySegmentStorage(
                                DataType.FLOAT32, 0, MemorySegment.ofArray(new byte[4]))),
                () -> assertThrows(
                        IllegalArgumentException.class,
                        () -> new MemorySegmentStorage(
                                DataType.FLOAT32, 1, MemorySegment.ofArray(new byte[3]))),
                () -> assertThrows(
                        IllegalArgumentException.class,
                        () -> new MemorySegmentStorage(
                                DataType.FLOAT32, 1, MemorySegment.ofArray(new byte[8]))),
                () -> assertThrows(
                        IllegalArgumentException.class,
                        () -> new MemorySegmentStorage(
                                DataType.FLOAT32, 1, MemorySegment.ofArray(new byte[5]))));
    }

    @Test
    void acceptsNativeConfinedSharedAndUnalignedSlicedSegments() {
        try (Arena confinedArena = Arena.ofConfined(); Arena sharedArena = Arena.ofShared()) {
            MemorySegment confined = confinedArena.allocate(16, 8);
            MemorySegment shared = sharedArena.allocate(16, 8);
            MemorySegment unaligned = confinedArena.allocate(8, 1).asSlice(1, 4);
            MemorySegment exactSlice = shared.asSlice(4, 8);

            MemorySegmentStorage confinedStorage =
                    new MemorySegmentStorage(DataType.INT64, 2, confined);
            MemorySegmentStorage sharedStorage =
                    new MemorySegmentStorage(DataType.FLOAT64, 2, shared);
            MemorySegmentStorage unalignedStorage =
                    new MemorySegmentStorage(DataType.FLOAT32, 1, unaligned);
            MemorySegmentStorage sliceStorage =
                    new MemorySegmentStorage(DataType.INT32, 2, exactSlice);

            assertAll(
                    () -> assertSame(confined, confinedStorage.segment()),
                    () -> assertSame(shared, sharedStorage.segment()),
                    () -> assertSame(unaligned, unalignedStorage.segment()),
                    () -> assertSame(exactSlice, sliceStorage.segment()),
                    () -> assertSame(confined.scope(), confinedStorage.segment().scope()),
                    () -> assertSame(shared.scope(), sharedStorage.segment().scope()));
        }
    }

    @Test
    void acceptsMappedSegmentAndRetainsItsIdentity() throws IOException {
        Path path = Files.createTempFile("synaptik-storage", ".bin");
        try {
            try (FileChannel channel = FileChannel.open(
                            path,
                            StandardOpenOption.READ,
                            StandardOpenOption.WRITE,
                            StandardOpenOption.TRUNCATE_EXISTING);
                    Arena arena = Arena.ofConfined()) {
                channel.truncate(8);
                MemorySegment mapped =
                        channel.map(FileChannel.MapMode.READ_WRITE, 0, 8, arena);
                MemorySegmentStorage storage =
                        new MemorySegmentStorage(DataType.INT64, 1, mapped);

                assertSame(mapped, storage.segment());
                assertTrue(mapped.isMapped());
                assertTrue(storage.isAlive());
            }
        } finally {
            Files.deleteIfExists(path);
        }
    }

    @Test
    void reportsReadOnlyAndPermitsRawMutationThroughWritableSegment() {
        MemorySegment writable = MemorySegment.ofArray(new byte[] {1, 2});
        MemorySegment readOnly = writable.asReadOnly();
        MemorySegmentStorage writableStorage =
                new MemorySegmentStorage(DataType.BOOL, 2, writable);
        MemorySegmentStorage readOnlyStorage =
                new MemorySegmentStorage(DataType.BOOL, 2, readOnly);

        writableStorage.segment().set(JAVA_BYTE, 1, (byte) 9);

        assertAll(
                () -> assertFalse(writableStorage.isReadOnly()),
                () -> assertTrue(readOnlyStorage.isReadOnly()),
                () -> assertSame(writable, writableStorage.segment()),
                () -> assertSame(readOnly, readOnlyStorage.segment()),
                () -> assertEquals(9, writable.get(JAVA_BYTE, 1)),
                () -> assertThrows(
                        IllegalArgumentException.class,
                        () -> readOnlyStorage.segment().set(JAVA_BYTE, 0, (byte) 4)));
    }

    @Test
    void rejectsInitiallyDeadScopeAfterSizeValidation() {
        Arena arena = Arena.ofConfined();
        MemorySegment exact = arena.allocate(4, 1);
        arena.close();

        IllegalStateException dead = assertThrows(
                IllegalStateException.class,
                () -> new MemorySegmentStorage(DataType.FLOAT32, 1, exact));
        IllegalArgumentException sizeWins = assertThrows(
                IllegalArgumentException.class,
                () -> new MemorySegmentStorage(DataType.FLOAT64, 1, exact));

        assertEquals("segment scope is not alive", dead.getMessage());
        assertEquals(
                "segment byte size must equal required byte size: required=8, actual=4",
                sizeWins.getMessage());
    }

    @Test
    void observesCallerClosureWithoutOwningOrReplacingSegment() {
        Arena arena = Arena.ofConfined();
        MemorySegment segment = arena.allocate(1, 1);
        MemorySegmentStorage storage =
                new MemorySegmentStorage(DataType.BOOL, 1, segment);

        assertTrue(storage.isAlive());
        arena.close();

        assertAll(
                () -> assertFalse(storage.isAlive()),
                () -> assertSame(segment, storage.segment()),
                () -> assertEquals(1, storage.byteSize()),
                () -> assertEquals(1, storage.elementCapacity()),
                () -> assertThrows(
                        IllegalStateException.class,
                        () -> storage.segment().get(JAVA_BYTE, 0)));
    }

    @Test
    void usesObjectIdentityAndInheritsObjectMethods() throws NoSuchMethodException {
        MemorySegment segment = MemorySegment.ofArray(new byte[1]);
        MemorySegmentStorage first =
                new MemorySegmentStorage(DataType.BOOL, 1, segment);
        MemorySegmentStorage second =
                new MemorySegmentStorage(DataType.BOOL, 1, segment);

        assertAll(
                () -> assertEquals(first, first),
                () -> assertNotEquals(first, second),
                () -> assertEquals(System.identityHashCode(first), first.hashCode()),
                () -> assertEquals(Object.class, first.getClass().getMethod("equals", Object.class).getDeclaringClass()),
                () -> assertEquals(Object.class, first.getClass().getMethod("hashCode").getDeclaringClass()),
                () -> assertEquals(Object.class, first.getClass().getMethod("toString").getDeclaringClass()),
                () -> assertTrue(first.toString().startsWith(MemorySegmentStorage.class.getName() + "@")));
    }
}
