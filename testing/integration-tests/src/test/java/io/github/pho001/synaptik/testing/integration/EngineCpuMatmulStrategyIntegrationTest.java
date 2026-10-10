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

/** Exact public Engine regression for a bare vector-eligible, default-scalar CPU MATMUL. */
final class EngineCpuMatmulStrategyIntegrationTest {
    @Test void defaultEngineRunsBareInt32MatmulWithoutVectorStrategy() {
        int m = 2, k = 63, n = 128;
        int[] left = new int[m * k], right = new int[k * n], expected = new int[m * n];
        for (int i = 0; i < left.length; i++) left[i] = i % 11 - 5;
        for (int i = 0; i < right.length; i++) right[i] = i % 13 - 6;
        for (int row = 0; row < m; row++) for (int column = 0; column < n; column++) {
            int sum = 0;
            for (int inner = 0; inner < k; inner++) {
                sum += left[row * k + inner] * right[inner * n + column];
            }
            expected[row * n + column] = sum;
        }
        try (Arena arena = Arena.ofShared(); Engine engine = Engine.standard()) {
            Tensor a = matrix(arena, Shape.of(m, k), left);
            Tensor b = matrix(arena, Shape.of(k, n), right);
            var compiled = engine.compile(List.of(a.matmul(b)));
            assertEquals(2, compiled.inputs().size());
            try (var prepared = engine.prepare(compiled);
                    var result = engine.run(prepared, List.of(a, b))) {
                var value = result.materialize(result.publications().getFirst(),
                        (long) expected.length * Integer.BYTES);
                var bytes = value.bytes();
                int[] actual = new int[expected.length];
                for (int i = 0; i < actual.length; i++) actual[i] = bytes.getInt();
                assertArrayEquals(expected, actual);
            }
        }
    }

    private static Tensor matrix(Arena arena, Shape shape, int[] values) {
        MemorySegment source = MemorySegment.ofArray(values);
        MemorySegment segment = arena.allocate(source.byteSize(), Integer.BYTES);
        MemorySegment.copy(source, 0, segment, 0, source.byteSize());
        var descriptor = new TensorDescriptor(DataType.INT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false);
        return TensorFactory.create(descriptor, Optional.empty(), Optional.of(
                new MemorySegmentStorage(DataType.INT32, values.length, segment)));
    }
}
