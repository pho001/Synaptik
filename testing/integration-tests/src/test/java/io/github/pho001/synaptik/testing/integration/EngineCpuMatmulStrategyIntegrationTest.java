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
import java.lang.foreign.ValueLayout;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/** Public Engine MATMUL regressions for default strategy and private/published transpose inputs. */
final class EngineCpuMatmulStrategyIntegrationTest {
    @Test void privateAndPublishedHeapTransposesSurviveSharedPrepareAndRepeatedRuns() {
        for (boolean publishView : List.of(false, true)) {
            float[] left = {1, 2, 3, 4, 5, 6};
            float[] weights = {7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
            try (var engine = Engine.standard()) {
                var a = heapMatrix(Shape.of(2, 3), left);
                var w = heapMatrix(Shape.of(4, 3), weights);
                Tensor transpose = w.permute(1, 0);
                Tensor product = a.matmul(transpose);
                var compiled = engine.compile(publishView ? List.of(transpose, product) : List.of(product));
                try (var prepared = engine.prepare(compiled)) {
                    for (int run = 0; run < 2; run++) {
                        if (run == 1) {
                            w.hostStorage().orElseThrow().segment()
                                    .setAtIndex(ValueLayout.JAVA_FLOAT, 0, 1);
                        }
                        try (var result = engine.run(prepared, List.of(a, w))) {
                            var output = result.materialize(result.publications().getLast(), 8L * Float.BYTES);
                            var bytes = output.bytes();
                            float[] actual = new float[8];
                            for (int i = 0; i < actual.length; i++) actual[i] = bytes.getFloat();
                            assertArrayEquals(run == 0
                                    ? new float[] {50, 68, 86, 104, 122, 167, 212, 257}
                                    : new float[] {44, 68, 86, 104, 98, 167, 212, 257}, actual);
                            if (publishView) {
                                var view = result.materialize(result.publications().getFirst(),
                                        12L * Float.BYTES).bytes();
                                float[] observed = new float[12];
                                for (int i = 0; i < observed.length; i++) observed[i] = view.getFloat();
                                assertArrayEquals(new float[] {run == 0 ? 7 : 1, 10, 13, 16,
                                        8, 11, 14, 17, 9, 12, 15, 18}, observed);
                            }
                        }
                    }
                }
            }
        }
    }

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

    /**
     * Imports exact values into independent heap storage that the test can update between runs.
     *
     * @param shape fully static matrix shape matching the flat values
     * @param values non-null row-major FLOAT32 array copied by the factory
     * @return tensor with a fresh heap copy, not an alias of the supplied array
     * @throws NullPointerException if shape or values is null
     * @throws IllegalArgumentException if shape and flat element count disagree
     */
    private static Tensor heapMatrix(Shape shape, float[] values) {
        return TensorFactory.fromFlatArray(new TensorDescriptor(DataType.FLOAT32, shape,
                Optional.of(LayoutDescriptor.contiguous(shape)), false), Optional.empty(), values);
    }
}
