package io.github.pho001.synaptik.testing.integration;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.model.datatype.DataType;
import io.github.pho001.synaptik.model.layout.LayoutDescriptor;
import io.github.pho001.synaptik.model.shape.Shape;
import io.github.pho001.synaptik.model.storage.MemorySegmentStorage;
import io.github.pho001.synaptik.model.tensor.Tensor;
import io.github.pho001.synaptik.model.tensor.TensorDescriptor;
import io.github.pho001.synaptik.model.tensor.TensorFactory;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/** Public CPU execution of a weight reused across two materialized matrix-product units. */
final class EngineCpuSharedExternalReadIntegrationTest {
    private static final Shape MATRIX = Shape.of(2, 2);
    private static final TensorDescriptor DESCRIPTOR = new TensorDescriptor(DataType.FLOAT32,
            MATRIX, Optional.of(LayoutDescriptor.contiguous(MATRIX)), false);

    @Test
    void sharedWeightChainsRunWithAndWithoutRelu() {
        try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
            Tensor left = matrix(arena, 1, -2, 3, 0.5f);
            Tensor weight = matrix(arena, 2, -1, -3, 1);
            float[] first = product(new float[] {1, -2, 3, 0.5f},
                    new float[] {2, -1, -3, 1});
            float[] chained = product(first, new float[] {2, -1, -3, 1});
            for (boolean relu : List.of(false, true)) {
                Tensor output = left.matmul(weight).matmul(weight);
                if (relu) output = output.relu();
                output = output.contiguous();
                float[] expected = chained.clone();
                if (relu) for (int i = 0; i < expected.length; i++) {
                    expected[i] = Math.max(0f, expected[i]);
                }
                assertArrayEquals(expected, run(engine, output, List.of(left, weight), 2));
            }
        }
    }

    @Test
    void distinctWeightChainAndSingleMatmulRemainExecutable() {
        try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
            float[] leftValues = {1, -2, 3, 0.5f};
            float[] firstWeightValues = {2, -1, -3, 1};
            float[] secondWeightValues = {1, 2, -1, 1};
            Tensor left = matrix(arena, leftValues);
            Tensor firstWeight = matrix(arena, firstWeightValues);
            Tensor secondWeight = matrix(arena, secondWeightValues);
            float[] first = product(leftValues, firstWeightValues);
            assertArrayEquals(first, run(engine, left.matmul(firstWeight).contiguous(),
                    List.of(left, firstWeight), 2));
            assertArrayEquals(product(first, secondWeightValues), run(engine,
                    left.matmul(firstWeight).matmul(secondWeight).contiguous(),
                    List.of(left, firstWeight, secondWeight), 3));
        }
    }

    private static float[] run(Engine engine, Tensor output, List<Tensor> inputs,
            int expectedInputCount) {
        var compiled = engine.compile(List.of(output));
        assertEquals(expectedInputCount, compiled.inputs().size());
        try (var prepared = engine.prepare(compiled);
                var result = engine.run(prepared, inputs)) {
            assertEquals(1, result.resultCount());
            var value = result.materialize(result.publications().getFirst(),
                    4L * Float.BYTES);
            var bytes = value.bytes();
            float[] actual = new float[4];
            for (int i = 0; i < actual.length; i++) actual[i] = bytes.getFloat();
            return actual;
        }
    }

    private static Tensor matrix(Arena arena, float... values) {
        MemorySegment source = MemorySegment.ofArray(values);
        MemorySegment segment = arena.allocate(source.byteSize(), Float.BYTES);
        MemorySegment.copy(source, 0, segment, 0, source.byteSize());
        return TensorFactory.create(DESCRIPTOR, Optional.empty(), Optional.of(
                new MemorySegmentStorage(DataType.FLOAT32, values.length, segment)));
    }

    private static float[] product(float[] left, float[] right) {
        float[] result = new float[4];
        for (int row = 0; row < 2; row++) {
            for (int column = 0; column < 2; column++) {
                float sum = 0f;
                for (int inner = 0; inner < 2; inner++) {
                    sum += left[row * 2 + inner] * right[inner * 2 + column];
                }
                result[row * 2 + column] = sum;
            }
        }
        return result;
    }
}
