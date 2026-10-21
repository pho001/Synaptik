package io.github.pho001.synaptik.text;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.data.PaddedSequenceBatch;
import io.github.pho001.synaptik.data.PaddingSide;
import io.github.pho001.synaptik.data.SequenceAxisOrder;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import java.lang.foreign.ValueLayout;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;

/** Text-to-Data adaptation, explicit configuration, geometry and output independence. */
final class TextSequenceBatcherTest {
    private static final WhitespaceTokenizer TOKENIZER = WhitespaceTokenizer.of(List.of("Hello", "world"));

    @Test
    void exposesOnlyTheSelectedAdapterAndBuilderApi() {
        assertTrue(Modifier.isFinal(TextSequenceBatcher.class.getModifiers()));
        assertEquals(0, TextSequenceBatcher.class.getConstructors().length);
        assertEquals(Set.of("builder", "batch"), Arrays.stream(TextSequenceBatcher.class.getDeclaredMethods())
                .filter(method -> Modifier.isPublic(method.getModifiers())).map(method -> method.getName())
                .collect(Collectors.toSet()));
        var builder = TextSequenceBatcher.Builder.class;
        assertTrue(Modifier.isPublic(builder.getModifiers()));
        assertTrue(Modifier.isStatic(builder.getModifiers()));
        assertTrue(Modifier.isFinal(builder.getModifiers()));
        assertEquals(0, builder.getConstructors().length);
        assertEquals(Set.of("leftPadded", "rightPadded", "timeMajor", "batchMajor", "build"),
                Arrays.stream(builder.getDeclaredMethods()).map(method -> method.getName()).collect(Collectors.toSet()));
    }

    @Test
    void requiresBothChoicesAndFreezesIndependentLastSelections() {
        assertThrows(IllegalStateException.class, () -> TextSequenceBatcher.builder(TOKENIZER).build());
        assertThrows(IllegalStateException.class, () -> TextSequenceBatcher.builder(TOKENIZER).leftPadded().build());
        assertThrows(IllegalStateException.class, () -> TextSequenceBatcher.builder(TOKENIZER).timeMajor().build());
        var builder = TextSequenceBatcher.builder(TOKENIZER);
        assertSame(builder, builder.leftPadded());
        assertSame(builder, builder.timeMajor());
        var first = builder.build();
        assertNotSame(first, builder.build());
        assertSame(builder, builder.rightPadded());
        assertSame(builder, builder.batchMajor());
        var texts = List.of("Hello world", "missing");
        assertArrayEquals(new long[] {2, 0, 3, 1}, cells(first.batch(texts)));
        assertArrayEquals(new long[] {2, 3, 1, 0}, cells(builder.build().batch(texts)));
    }

    @Test
    void preservesIdsCountsAndGeometryForAllFourConfigurations() {
        long[][] encoded = {{2, 3, 2}, {1}, {}, {3, 1}};
        for (var side : PaddingSide.values()) for (var order : SequenceAxisOrder.values()) {
            var batch = configured(side, order).batch(List.of("Hello world Hello", "missing", " \t", "world absent"));
            assertArrayEquals(new long[] {3, 1, 0, 2}, batch.validLengths().toLongArray());
            assertSame(side, batch.paddingSide());
            assertSame(order, batch.axisOrder());
            Shape shape = order == SequenceAxisOrder.TIME_MAJOR ? Shape.of(3, 4) : Shape.of(4, 3);
            var tensor = batch.values();
            assertEquals(shape, tensor.descriptor().shape());
            assertEquals(DataType.INT64, tensor.descriptor().dataType());
            assertEquals(LayoutDescriptor.contiguous(shape), tensor.descriptor().layout().orElseThrow());
            assertFalse(tensor.descriptor().requiresGrad());
            assertTrue(tensor.provenance().isEmpty());
            assertTrue(tensor.hostStorage().orElseThrow().segment().heapBase().isPresent());
            long[] actual = cells(batch);
            for (int row = 0; row < 4; row++) for (int time = 0; time < 3; time++) {
                int source = side == PaddingSide.LEFT ? time - (3 - encoded[row].length) : time;
                long expected = source >= 0 && source < encoded[row].length ? encoded[row][source] : 0;
                int index = order == SequenceAxisOrder.TIME_MAJOR ? time * 4 + row : row * 3 + time;
                assertEquals(expected, actual[index]);
            }
        }
    }

    @Test
    void handlesEmptyOuterAndAllEmptyRowsInEveryConfiguration() {
        for (var side : PaddingSide.values()) for (var order : SequenceAxisOrder.values()) {
            for (var texts : List.of(List.<String>of(), List.of("", " \u2003"))) {
                var batch = configured(side, order).batch(texts);
                assertEquals(order == SequenceAxisOrder.TIME_MAJOR ? Shape.of(0, texts.size())
                        : Shape.of(texts.size(), 0), batch.values().descriptor().shape());
                assertArrayEquals(new long[texts.size()], batch.validLengths().toLongArray());
                assertArrayEquals(new long[0], cells(batch));
                assertNotSame(batch.values(), configured(side, order).batch(texts).values());
            }
        }
    }

    @Test
    void rejectsNamedNullsWithoutPoisoningTheConfiguration() {
        assertEquals("tokenizer", assertThrows(NullPointerException.class,
                () -> TextSequenceBatcher.builder(null)).getMessage());
        var batcher = configured(PaddingSide.RIGHT, SequenceAxisOrder.BATCH_MAJOR);
        assertEquals("texts", assertThrows(NullPointerException.class, () -> batcher.batch(null)).getMessage());
        assertTrue(assertThrows(NullPointerException.class,
                () -> batcher.batch(Arrays.asList("Hello", null))).getMessage().contains("row 1"));
        assertArrayEquals(new long[] {2}, cells(batcher.batch(List.of("Hello"))));
    }

    @Test
    void sequentialInputsAndRepeatedOutputsAreIndependent() {
        var texts = new LinkedList<>(List.of("Hello world", "missing")) {
            @Override public String get(int index) { throw new AssertionError("indexed traversal"); }
        };
        var batcher = configured(PaddingSide.LEFT, SequenceAxisOrder.BATCH_MAJOR);
        var first = batcher.batch(texts);
        var second = batcher.batch(texts);
        texts.clear();
        assertNotSame(first.values(), second.values());
        assertNotSame(first.values().hostStorage().orElseThrow(), second.values().hostStorage().orElseThrow());
        assertArrayEquals(new long[] {2, 3, 0, 1}, cells(first));
        first.values().hostStorage().orElseThrow().segment().setAtIndex(ValueLayout.JAVA_LONG, 0, 99);
        assertArrayEquals(new long[] {2, 3, 0, 1}, cells(second));
        assertArrayEquals(new long[] {2, 1}, first.validLengths().toLongArray());
    }

    /**
     * Creates an explicitly selected adapter without introducing a default in production.
     * @param side non-null padding interval choice
     * @param order non-null leading-axis choice
     * @return a fresh configured adapter
     */
    private static TextSequenceBatcher configured(PaddingSide side, SequenceAxisOrder order) {
        var builder = TextSequenceBatcher.builder(TOKENIZER);
        if (side == PaddingSide.LEFT) builder.leftPadded(); else builder.rightPadded();
        if (order == SequenceAxisOrder.TIME_MAJOR) builder.timeMajor(); else builder.batchMajor();
        return builder.build();
    }

    /**
     * Reads the contiguous heap output in physical element order.
     * @param batch non-null batch created by Text
     * @return a fresh array of all stored IDs
     */
    private static long[] cells(PaddedSequenceBatch batch) {
        return batch.values().hostStorage().orElseThrow().segment().toArray(ValueLayout.JAVA_LONG);
    }
}
