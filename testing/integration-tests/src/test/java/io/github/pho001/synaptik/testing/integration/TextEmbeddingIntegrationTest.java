package io.github.pho001.synaptik.testing.integration;

import static org.junit.jupiter.api.Assertions.*;

import io.github.pho001.synaptik.data.PaddingSide;
import io.github.pho001.synaptik.data.SequenceAxisOrder;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import io.github.pho001.synaptik.nn.layers.Embedding;
import io.github.pho001.synaptik.text.TextSequenceBatcher;
import io.github.pho001.synaptik.text.WhitespaceTokenizer;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/** Composes Text and unchanged Embedding externally and verifies actual Engine results. */
final class TextEmbeddingIntegrationTest {
    /** Checks every physical output element, including ordinary nonzero PAD row gathers. */
    @Test
    void gathersKnownUnknownAndNonzeroPaddingRowsForEverySideAndOrder() {
        var tokenizer = WhitespaceTokenizer.of(List.of("Hello", "world"));
        float[] table = {11.5f, -7.25f, 21.0f, 22.0f, 31.0f, 32.0f, 41.0f, 42.0f};
        long[][] ids = {{2, 3, 2}, {1}, {}, {3, 1}};
        try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
            Shape weightShape = Shape.of(tokenizer.vocabularySize(), 2);
            MemorySegment segment = arena.allocate((long) table.length * Float.BYTES, Float.BYTES);
            segment.copyFrom(MemorySegment.ofArray(table));
            var weight = TensorFactory.create(new TensorDescriptor(DataType.FLOAT32, weightShape,
                    Optional.of(LayoutDescriptor.contiguous(weightShape)), true), Optional.empty(),
                    Optional.of(new MemorySegmentStorage(DataType.FLOAT32, table.length, segment)));
            var embedding = new Embedding(weight);
            for (var side : PaddingSide.values()) for (var order : SequenceAxisOrder.values()) {
                var builder = TextSequenceBatcher.builder(tokenizer);
                if (side == PaddingSide.LEFT) builder.leftPadded(); else builder.rightPadded();
                if (order == SequenceAxisOrder.TIME_MAJOR) builder.timeMajor(); else builder.batchMajor();
                var batch = builder.build().batch(List.of("Hello world Hello", "missing", "", "world absent"));
                assertArrayEquals(new long[] {3, 1, 0, 2}, batch.validLengths().toLongArray());
                assertSame(side, batch.paddingSide());
                assertSame(order, batch.axisOrder());
                var actual = engine.compute(embedding.forward(batch.values()));
                assertEquals(DataType.FLOAT32, actual.dataType());
                assertEquals(order == SequenceAxisOrder.TIME_MAJOR ? Shape.of(3, 4, 2) : Shape.of(4, 3, 2), actual.shape());
                float[] expected = new float[24];
                long[] input = batch.values().hostStorage().orElseThrow().segment().toArray(ValueLayout.JAVA_LONG);
                int paddedPositions = 0;
                for (int row = 0; row < ids.length; row++) for (int time = 0; time < 3; time++) {
                    int source = side == PaddingSide.LEFT ? time - (3 - ids[row].length) : time;
                    boolean padding = source < 0 || source >= ids[row].length;
                    int id = padding ? 0 : Math.toIntExact(ids[row][source]);
                    int position = order == SequenceAxisOrder.TIME_MAJOR ? time * 4 + row : row * 3 + time;
                    assertEquals(id, input[position]);
                    expected[position * 2] = table[id * 2];
                    expected[position * 2 + 1] = table[id * 2 + 1];
                    if (padding) {
                        paddedPositions++;
                        assertEquals(11.5f, actual.bytes().getFloat(position * 2 * Float.BYTES));
                        assertEquals(-7.25f, actual.bytes().getFloat((position * 2 + 1) * Float.BYTES));
                    }
                }
                float[] values = new float[expected.length];
                actual.bytes().asFloatBuffer().get(values);
                assertArrayEquals(expected, values);
                assertEquals(6, paddedPositions);
            }
        }
    }
}
