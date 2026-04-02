import backend.runtime.ExecutionMode;
import config.optimizer.OptimizerConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
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
        CompiledGraph.compile(reshaped, OptimizerConfig.noOptimization()).execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
        assertArrayEquals(new int[]{3, 2}, reshaped.getShape());
        assertArrayEquals(new double[]{1, 2, 3, 4, 5, 6}, reshaped.toDoubleArrayCopy(), eps(dataType));

        Tensor expanded = reshaped.expandDims(1);
        CompiledGraph.compile(expanded, OptimizerConfig.noOptimization()).execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
        assertArrayEquals(new int[]{3, 1, 2}, expanded.getShape());
        assertArrayEquals(new double[]{1, 2, 3, 4, 5, 6}, expanded.toDoubleArrayCopy(), eps(dataType));

        Tensor squeezed = expanded.squeeze(1);
        CompiledGraph.compile(squeezed, OptimizerConfig.noOptimization()).execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
        assertArrayEquals(new int[]{3, 2}, squeezed.getShape());
        assertArrayEquals(new double[]{1, 2, 3, 4, 5, 6}, squeezed.toDoubleArrayCopy(), eps(dataType));
    }

    @ParameterizedTest
    @EnumSource(value = DataType.class, names = {"FLOAT32", "FLOAT64"})
    void expandBroadcastsSingletonDimensions(DataType dataType) {
        Tensor base = new Tensor(new double[]{1, 2, 3}, new int[]{1, 3}, null, "base", dataType);
        Tensor expanded = base.expand(2, 3);
        CompiledGraph.compile(expanded, OptimizerConfig.noOptimization()).execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
        assertArrayEquals(new int[]{2, 3}, expanded.getShape());
        assertArrayEquals(new int[]{0, 1}, expanded.getStrides());
        assertSame(base.getStorage(), expanded.getStorage());
        assertArrayEquals(new double[]{1, 2, 3, 1, 2, 3}, expanded.toDoubleArrayCopy(), eps(dataType));
    }

    @Test
    void expandSupportsLeadingRankExpansion() {
        Tensor base = new Tensor(new double[]{1, 2, 3}, new int[]{3}, null, "base", DataType.FLOAT64);
        Tensor expanded = base.expand(2, 3);
        CompiledGraph.compile(expanded, OptimizerConfig.noOptimization()).execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
        assertArrayEquals(new int[]{2, 3}, expanded.getShape());
        assertArrayEquals(new int[]{0, 1}, expanded.getStrides());
        assertSame(base.getStorage(), expanded.getStorage());
        assertArrayEquals(new double[]{1, 2, 3, 1, 2, 3}, expanded.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void expandContiguousMaterializesDenseCopy() {
        Tensor base = new Tensor(new double[]{1, 2, 3}, new int[]{1, 3}, null, "base", DataType.FLOAT64);
        Tensor expanded = base.expand(2, 3);
        Tensor materialized = expanded.contiguous();

        CompiledGraph.compile(materialized, OptimizerConfig.noOptimization()).execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new int[]{2, 3}, materialized.getShape());
        assertArrayEquals(new int[]{3, 1}, materialized.getStrides());
        assertArrayEquals(new double[]{1, 2, 3, 1, 2, 3}, materialized.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void expandBackwardReducesToOriginalShape() {
        Tensor base = new Tensor(new double[]{1, 2, 3}, new int[]{1, 3}, null, "base", DataType.FLOAT64);
        base.setRequiresGrad(true);
        Tensor expanded = base.expand(2, 3);
        CompiledGraph.compile(expanded, OptimizerConfig.trainingDefaults()).execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);
        assertArrayEquals(new int[]{1, 3}, base.getGradient().getShape());
        assertArrayEquals(new double[]{2.0, 2.0, 2.0}, base.getGradient().toDoubleArrayCopy(), 1e-9);
    }

    @ParameterizedTest
    @EnumSource(value = DataType.class, names = {"FLOAT32", "FLOAT64"})
    void permuteAndTransposeDataAndShape(DataType dataType) {
        Tensor base = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "base", dataType);

        Tensor permuted = base.permute(1, 0);
        CompiledGraph.compile(permuted, OptimizerConfig.noOptimization()).execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
        assertArrayEquals(new int[]{3, 2}, permuted.getShape());
        assertSame(base.getStorage(), permuted.getStorage());
        Tensor permutedContiguous = permuted.contiguous();
        CompiledGraph.compile(permutedContiguous, OptimizerConfig.noOptimization()).execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
        assertArrayEquals(new double[]{1, 4, 2, 5, 3, 6}, permutedContiguous.toDoubleArrayCopy(), eps(dataType));

        Tensor transposed = base.transpose();
        CompiledGraph.compile(transposed, OptimizerConfig.noOptimization()).execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
        assertArrayEquals(new int[]{3, 2}, transposed.getShape());
        Tensor transposedContiguous = transposed.contiguous();
        CompiledGraph.compile(transposedContiguous, OptimizerConfig.noOptimization()).execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
        assertArrayEquals(new double[]{1, 4, 2, 5, 3, 6}, transposedContiguous.toDoubleArrayCopy(), eps(dataType));
    }

    @Test
    void reshapeWithInferredDimension() {
        Tensor base = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "base", DataType.FLOAT32);
        Tensor reshaped = base.reshape(3, -1);
        CompiledGraph.compile(reshaped, OptimizerConfig.noOptimization()).execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
        assertArrayEquals(new int[]{3, 2}, reshaped.getShape());
        assertArrayEquals(new double[]{1, 2, 3, 4, 5, 6}, reshaped.toDoubleArrayCopy(), 1e-6);
    }

    @Test
    void reshapeRejectsInvalidSize() {
        Tensor base = new Tensor(new double[]{1, 2, 3, 4}, new int[]{2, 2}, null, "base", DataType.FLOAT32);
        assertThrows(IllegalArgumentException.class, () -> base.reshape(3, 2));
    }

    @Test
    void expandRejectsNonSingletonExpansion() {
        Tensor base = new Tensor(new double[]{1, 2, 3, 4}, new int[]{2, 2}, null, "base", DataType.FLOAT32);
        assertThrows(IllegalArgumentException.class, () -> base.expand(2, 3));
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
