package io.github.pho001.synaptik.testing.integration;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.pho001.synaptik.backend.cpu.CpuBackendIntegration;
import io.github.pho001.synaptik.engine.Engine;
import io.github.pho001.synaptik.engine.HostTensorValue;
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

/** Verifies exact extraction closure through public reusable CPU execution and scalar backward. */
final class CompilerSelectSliceLayoutIntegrationTest {
    @Test
    void selectsLastBatchedProductWithReusableRunsAndIndependentNonzeroGradients() {
        exercise(true, 2);
    }

    @Test
    void slicesNonzeroGruAndLstmGatesWithReusableRunsAndIndependentNonzeroGradients() {
        exercise(false, 6);
        exercise(false, 8);
    }

    /**
     * Compares an extracted matrix product and both input gradients with small-array arithmetic.
     *
     * @param select whether to remove the last batch coordinate instead of slicing columns
     * @param columns product width; two for selection or the six/eight gate projection width
     */
    private static void exercise(boolean select, int columns) {
        int rows = select ? 6 : 2;
        float[] left = new float[rows * 2];
        float[] right = new float[2 * columns];
        for (int index = 0; index < left.length; index++) left[index] = index + 1;
        for (int index = 0; index < right.length; index++) right[index] = index % 5 - 2;
        float[] expected = new float[4];
        float[] leftGradient = new float[left.length];
        float[] rightGradient = new float[right.length];
        float objective = 0;
        for (int row = 0; row < 2; row++) {
            int sourceRow = select ? row + 4 : row;
            for (int column = 0; column < 2; column++) {
                int sourceColumn = select ? column : column + 2;
                float value = 0;
                for (int inner = 0; inner < 2; inner++) {
                    value += left[sourceRow * 2 + inner] * right[inner * columns + sourceColumn];
                }
                expected[row * 2 + column] = value;
                objective += value * value;
                for (int inner = 0; inner < 2; inner++) {
                    leftGradient[sourceRow * 2 + inner] += 2 * value * right[inner * columns + sourceColumn];
                    rightGradient[inner * columns + sourceColumn] += 2 * value * left[sourceRow * 2 + inner];
                }
            }
        }
        assertTrue(hasNonzero(leftGradient));
        assertTrue(hasNonzero(rightGradient));

        try (Arena arena = Arena.ofConfined();
                Engine engine = Engine.builder().takeOwnership(CpuBackendIntegration.open()).build()) {
            Tensor input = input(arena, select ? Shape.of(3, 2, 2) : Shape.of(2, 2), left);
            Tensor weights = input(arena, Shape.of(2, columns), right);
            Tensor product = input.matmul(weights);
            Tensor extracted = select ? product.select(0, 2) : product.sliceAxis(1, 2, 4);
            var compiled = engine.compile(List.of(extracted));
            try (var prepared = engine.prepare(compiled)) {
                for (int run = 0; run < 2; run++) {
                    try (var result = engine.run(prepared, List.of(input, weights))) {
                        assertArrayEquals(expected, floats(result.materialize(
                                result.publications().getFirst(), 4L * Float.BYTES)));
                    }
                }
            }
            var backward = engine.backward(extracted.mul(extracted).sum(), List.of(input, weights),
                    (1L + left.length + right.length) * Float.BYTES);
            assertEquals(objective, floats(backward.objective())[0]);
            assertArrayEquals(leftGradient, floats(backward.gradients().get(0)));
            assertArrayEquals(rightGradient, floats(backward.gradients().get(1)));
        }
    }

    /**
     * Checks that a gradient oracle exercises at least one addressed cell.
     *
     * @param values non-null dense expected gradient values
     * @return whether any expected derivative is nonzero
     */
    private static boolean hasNonzero(float[] values) {
        for (float value : values) if (value != 0) return true;
        return false;
    }

    /**
     * Copies a detached publication into the assertion's dense array.
     *
     * @param value non-null FLOAT32 host publication
     * @return a fresh array in logical publication order
     */
    private static float[] floats(HostTensorValue value) {
        float[] result = new float[Math.toIntExact(value.elementCount())];
        value.bytes().asFloatBuffer().get(result);
        return result;
    }

    /**
     * Creates a gradient-bearing caller leaf with arena-owned canonical FLOAT32 storage.
     *
     * @param arena non-null open confined arena owning the copied input storage
     * @param shape non-null fully static shape with exactly {@code values.length} elements
     * @param values non-null dense input values, copied without retaining their array
     * @return a fresh leaf borrowing storage until the arena closes
     */
    private static Tensor input(Arena arena, Shape shape, float[] values) {
        MemorySegment source = MemorySegment.ofArray(values);
        MemorySegment storage = arena.allocate(source.byteSize(), Float.BYTES);
        MemorySegment.copy(source, 0, storage, 0, source.byteSize());
        return TensorFactory.create(new TensorDescriptor(DataType.FLOAT32, shape,
                        Optional.of(LayoutDescriptor.contiguous(shape)), true),
                Optional.empty(), Optional.of(new MemorySegmentStorage(DataType.FLOAT32, values.length, storage)));
    }
}
