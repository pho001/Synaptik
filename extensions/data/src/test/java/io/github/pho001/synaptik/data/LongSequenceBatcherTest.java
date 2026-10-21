package io.github.pho001.synaptik.data;

import static org.junit.jupiter.api.Assertions.*;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import java.lang.foreign.ValueLayout;
import java.lang.reflect.Modifier;
import java.util.AbstractList;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;

/** Bounded allocation, geometry, configuration and ownership checks for integral batching. */
final class LongSequenceBatcherTest {
    @Test
    void exposesOnlyTheApprovedConfigurationApi() {
        assertTrue(Modifier.isFinal(LongSequenceBatcher.class.getModifiers()));
        assertEquals(0, LongSequenceBatcher.class.getConstructors().length);
        assertEquals(Set.of("builder", "batch"), Arrays.stream(LongSequenceBatcher.class.getDeclaredMethods())
                .filter(method -> Modifier.isPublic(method.getModifiers())).map(method -> method.getName())
                .collect(Collectors.toSet()));
        var builder = LongSequenceBatcher.Builder.class;
        assertTrue(Modifier.isPublic(builder.getModifiers()));
        assertTrue(Modifier.isStatic(builder.getModifiers()));
        assertTrue(Modifier.isFinal(builder.getModifiers()));
        assertEquals(0, builder.getConstructors().length);
        assertEquals(Set.of("paddingValue", "leftPadded", "rightPadded", "timeMajor", "batchMajor", "build"),
                Arrays.stream(builder.getDeclaredMethods()).map(method -> method.getName()).collect(Collectors.toSet()));
    }

    @Test
    void requiresEveryChoiceIncludingExplicitZeroAndFreezesIndependentConfigurations() {
        for (int mask = 0; mask < 7; mask++) {
            var builder = LongSequenceBatcher.builder();
            if ((mask & 1) != 0) builder.paddingValue(0);
            if ((mask & 2) != 0) builder.leftPadded();
            if ((mask & 4) != 0) builder.timeMajor();
            assertThrows(IllegalStateException.class, builder::build);
        }
        var builder = LongSequenceBatcher.builder();
        assertSame(builder, builder.paddingValue(0));
        assertSame(builder, builder.leftPadded());
        assertSame(builder, builder.timeMajor());
        var first = builder.build();
        assertNotSame(first, builder.build());
        assertSame(builder, builder.paddingValue(9));
        assertSame(builder, builder.rightPadded());
        assertSame(builder, builder.batchMajor());
        var rows = List.of(new long[] {1, 2}, new long[] {3});
        assertArrayEquals(new long[] {1, 0, 2, 3}, cells(first.batch(rows)));
        assertArrayEquals(new long[] {1, 2, 3, 9}, cells(builder.build().batch(rows)));
    }

    @Test
    void preservesEveryValueAndRowAcrossBothSidesAndOrdersWithExtremePadding() {
        for (var side : PaddingSide.values()) for (var order : SequenceAxisOrder.values()) {
            for (long pad : new long[] {0, -7, Long.MIN_VALUE, Long.MAX_VALUE}) {
                var rows = List.of(new long[] {pad, -4, Long.MAX_VALUE}, new long[] {Long.MIN_VALUE},
                        new long[0], new long[] {8, 9});
                var batch = configured(side, order, pad).batch(rows);
                assertArrayEquals(new long[] {3, 1, 0, 2}, batch.validLengths().toLongArray());
                assertSame(side, batch.paddingSide());
                assertSame(order, batch.axisOrder());
                Shape shape = order == SequenceAxisOrder.TIME_MAJOR ? Shape.of(3, 4) : Shape.of(4, 3);
                var values = batch.values();
                assertEquals(shape, values.descriptor().shape());
                assertEquals(DataType.INT64, values.descriptor().dataType());
                assertFalse(values.descriptor().requiresGrad());
                assertTrue(values.provenance().isEmpty());
                assertEquals(LayoutDescriptor.contiguous(shape), values.descriptor().layout().orElseThrow());
                var storage = values.hostStorage().orElseThrow();
                assertEquals(12, storage.elementCapacity());
                assertEquals(96, storage.byteSize());
                assertFalse(storage.isReadOnly());
                assertTrue(storage.isAlive());
                assertTrue(storage.segment().heapBase().isPresent());
                long[] actual = cells(batch);
                for (int row = 0; row < 4; row++) for (int step = 0; step < 3; step++) {
                    int source = side == PaddingSide.LEFT ? step - (3 - rows.get(row).length) : step;
                    long expected = source >= 0 && source < rows.get(row).length ? rows.get(row)[source] : pad;
                    int index = order == SequenceAxisOrder.TIME_MAJOR ? step * 4 + row : row * 3 + step;
                    assertEquals(expected, actual[index]);
                }
            }
        }
    }

    @Test
    void emptyAndAllEmptyInputsHaveOrdinaryEmptyHeapStorageForEveryChoice() {
        for (var side : PaddingSide.values()) for (var order : SequenceAxisOrder.values()) {
            for (var rows : List.of(List.<long[]>of(), List.of(new long[0], new long[0]))) {
                var batch = configured(side, order, Long.MIN_VALUE).batch(rows);
                assertEquals(order == SequenceAxisOrder.TIME_MAJOR ? Shape.of(0, rows.size())
                        : Shape.of(rows.size(), 0), batch.values().descriptor().shape());
                assertArrayEquals(new long[rows.size()], batch.validLengths().toLongArray());
                assertArrayEquals(new long[0], cells(batch));
                assertTrue(batch.values().hostStorage().orElseThrow().segment().heapBase().isPresent());
            }
        }
    }

    @Test
    void rejectsNamedNullsAndRemainsReusable() {
        var batcher = configured(PaddingSide.RIGHT, SequenceAxisOrder.BATCH_MAJOR, 0);
        assertEquals("sequences", assertThrows(NullPointerException.class, () -> batcher.batch(null)).getMessage());
        for (int row = 0; row < 2; row++) {
            var rows = new ArrayList<>(List.of(new long[] {1}, new long[] {2}));
            rows.set(row, null);
            String message = assertThrows(NullPointerException.class, () -> batcher.batch(rows)).getMessage();
            assertTrue(message.contains("row " + row));
        }
        assertArrayEquals(new long[] {1}, cells(batcher.batch(List.of(new long[] {1}))));
    }

    @Test
    void inputsAndRepeatedOutputsAreIndependentAfterReturn() {
        long[] first = {1, 2};
        var rows = new ArrayList<>(List.of(first, new long[] {3}));
        var batcher = configured(PaddingSide.LEFT, SequenceAxisOrder.BATCH_MAJOR, 7);
        var a = batcher.batch(rows);
        var b = batcher.batch(rows);
        assertNotSame(a.values(), b.values());
        assertNotSame(a.values().hostStorage().orElseThrow(), b.values().hostStorage().orElseThrow());
        first[0] = 99;
        rows.set(1, new long[0]);
        rows.clear();
        assertArrayEquals(new long[] {1, 2, 7, 3}, cells(a));
        a.values().hostStorage().orElseThrow().segment().setAtIndex(ValueLayout.JAVA_LONG, 0, 88);
        assertArrayEquals(new long[] {1, 2, 7, 3}, cells(b));
        assertArrayEquals(new long[] {2, 1}, a.validLengths().toLongArray());
    }

    @Test
    void linkedListsNeedNoIndexedAccessInEitherPassAndRetainRowDiagnostics() {
        var rows = new LinkedList<long[]>() {
            /**
             * Rejects random access so both batch passes must use sequential traversal.
             * @param index requested row, never supported by this fixture
             * @return never returns
             * @throws AssertionError on any indexed access
             */
            @Override public long[] get(int index) {
                throw new AssertionError("Indexed access prohibited at row " + index);
            }
        };
        var expectedRows = List.of(new long[] {1, 2}, new long[0], new long[] {3});
        rows.addAll(expectedRows);
        for (var side : PaddingSide.values()) for (var order : SequenceAxisOrder.values()) {
            var batcher = configured(side, order, -7);
            var actual = batcher.batch(rows);
            assertArrayEquals(cells(batcher.batch(expectedRows)), cells(actual));
            assertArrayEquals(new long[] {2, 0, 1}, actual.validLengths().toLongArray());
            assertSame(side, actual.paddingSide());
            assertSame(order, actual.axisOrder());
        }
        rows.add(null);
        assertEquals("sequences row 3", assertThrows(NullPointerException.class,
                () -> configured(PaddingSide.RIGHT, SequenceAxisOrder.BATCH_MAJOR, 0).batch(rows)).getMessage());
    }

    @Test
    void enormousVirtualListsRejectAtFirstRowBeforeBatchSizedAllocationOrLaterReads() {
        for (int length : new int[] {2, 4}) {
            var rows = new AbstractList<long[]>() {
                @Override public int size() { return Integer.MAX_VALUE; }
                @Override public long[] get(int index) {
                    assertEquals(0, index, "oversized geometry must reject before the next row");
                    return new long[length];
                }
            };
            for (var side : PaddingSide.values()) for (var order : SequenceAxisOrder.values()) {
                assertThrows(IllegalArgumentException.class, () -> configured(side, order, 0).batch(rows));
            }
        }
    }

    /**
     * Builds an explicit test configuration.
     * @param side non-null padding side
     * @param order non-null logical order
     * @param pad exact padding value
     * @return fresh configured batcher
     */
    private static LongSequenceBatcher configured(PaddingSide side, SequenceAxisOrder order, long pad) {
        var builder = LongSequenceBatcher.builder().paddingValue(pad);
        if (side == PaddingSide.LEFT) builder.leftPadded(); else builder.rightPadded();
        if (order == SequenceAxisOrder.TIME_MAJOR) builder.timeMajor(); else builder.batchMajor();
        return builder.build();
    }

    /**
     * Reads a small batch in physical contiguous order.
     * @param batch non-null live heap batch
     * @return fresh flat long array
     */
    private static long[] cells(PaddedSequenceBatch batch) {
        return batch.values().hostStorage().orElseThrow().segment().toArray(ValueLayout.JAVA_LONG);
    }
}
