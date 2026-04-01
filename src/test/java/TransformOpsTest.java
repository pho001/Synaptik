import graph.optimizer.GraphOptimizer;
import tensor.DataType;
import tensor.Tensor;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class TransformOpsTest {

    @ParameterizedTest
    @EnumSource(value = DataType.class, names = {"FLOAT32", "FLOAT64"})
    void reshapeExpandSqueezeRoundTrip(DataType dataType) {
        Tensor base = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "base", dataType);

        Tensor reshaped = base.reshape(3, 2);
        TestGraphSupport.execute(reshaped, new GraphOptimizer());
        assertArrayEquals(new int[]{3, 2}, reshaped.getShape());
        assertArrayEquals(new double[]{1, 2, 3, 4, 5, 6}, reshaped.toDoubleArrayCopy(), eps(dataType));

        Tensor expanded = reshaped.expandDims(1);
        TestGraphSupport.execute(expanded, new GraphOptimizer());
        assertArrayEquals(new int[]{3, 1, 2}, expanded.getShape());
        assertArrayEquals(new double[]{1, 2, 3, 4, 5, 6}, expanded.toDoubleArrayCopy(), eps(dataType));

        Tensor squeezed = expanded.squeeze(1);
        TestGraphSupport.execute(squeezed, new GraphOptimizer());
        assertArrayEquals(new int[]{3, 2}, squeezed.getShape());
        assertArrayEquals(new double[]{1, 2, 3, 4, 5, 6}, squeezed.toDoubleArrayCopy(), eps(dataType));
    }

    @ParameterizedTest
    @EnumSource(value = DataType.class, names = {"FLOAT32", "FLOAT64"})
    void permuteAndTransposeDataAndShape(DataType dataType) {
        Tensor base = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "base", dataType);

        Tensor permuted = base.permute(1, 0);
        TestGraphSupport.execute(permuted, new GraphOptimizer());
        assertArrayEquals(new int[]{3, 2}, permuted.getShape());
        assertSame(base.getStorage(), permuted.getStorage());
        Tensor permutedContiguous = permuted.contiguous();
        TestGraphSupport.execute(permutedContiguous, new GraphOptimizer());
        assertArrayEquals(new double[]{1, 4, 2, 5, 3, 6}, permutedContiguous.toDoubleArrayCopy(), eps(dataType));

        Tensor transposed = base.transpose();
        TestGraphSupport.execute(transposed, new GraphOptimizer());
        assertArrayEquals(new int[]{3, 2}, transposed.getShape());
        Tensor transposedContiguous = transposed.contiguous();
        TestGraphSupport.execute(transposedContiguous, new GraphOptimizer());
        assertArrayEquals(new double[]{1, 4, 2, 5, 3, 6}, transposedContiguous.toDoubleArrayCopy(), eps(dataType));
    }

    @Test
    void reshapeWithInferredDimension() {
        Tensor base = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "base", DataType.FLOAT32);
        Tensor reshaped = base.reshape(3, -1);
        TestGraphSupport.execute(reshaped, new GraphOptimizer());
        assertArrayEquals(new int[]{3, 2}, reshaped.getShape());
        assertArrayEquals(new double[]{1, 2, 3, 4, 5, 6}, reshaped.toDoubleArrayCopy(), 1e-6);
    }

    @Test
    void reshapeRejectsInvalidSize() {
        Tensor base = new Tensor(new double[]{1, 2, 3, 4}, new int[]{2, 2}, null, "base", DataType.FLOAT32);
        assertThrows(IllegalArgumentException.class, () -> base.reshape(3, 2));
    }

    @Test
    void squeezeRejectsNonSingletonAxis() {
        Tensor base = new Tensor(new double[]{1, 2, 3, 4}, new int[]{2, 2}, null, "base", DataType.FLOAT32);
        assertThrows(IllegalArgumentException.class, () -> base.squeeze(1));
    }

    @Test
    void transposeRequiresRankTwo() {
        Tensor base = new Tensor(new double[]{1, 2, 3, 4}, new int[]{2, 2, 1}, null, "base", DataType.FLOAT32);
        Exception ex = assertThrows(IllegalStateException.class, base::transpose);
        assertEquals("transpose() requires rank-2 tensor, got rank=3", ex.getMessage());
    }

    private static double eps(DataType dataType) {
        return dataType == DataType.FLOAT64 ? 1e-12 : 1e-6;
    }
}
