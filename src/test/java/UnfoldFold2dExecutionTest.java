import backend.runtime.ExecutionMode;
import config.compile.CompileConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.options.Window2dOptions;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class UnfoldFold2dExecutionTest {
    @Test
    void unfold2dMaterializesIm2colColumns() {
        Tensor input = new Tensor(new double[]{
                1, 2, 3,
                4, 5, 6,
                7, 8, 9
        }, new int[]{1, 1, 3, 3}, null, "input", DataType.FLOAT64);
        Tensor out = input.unfold2d(Window2dOptions.of(2, 2));

        CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults())
                .execute(ExecutionMode.FORWARD);

        assertArrayEquals(new int[]{1, 4, 4}, out.getShape());
        assertArrayEquals(new double[]{
                1, 2, 4, 5,
                2, 3, 5, 6,
                4, 5, 7, 8,
                5, 6, 8, 9
        }, out.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void unfold2dAppliesPaddingStrideAndDilation() {
        Tensor input = new Tensor(new double[]{
                1, 2,
                3, 4
        }, new int[]{1, 1, 2, 2}, null, "input", DataType.FLOAT32);
        Window2dOptions options = new Window2dOptions(2, 2, 1, 1, 1, 1, 1, 1);
        Tensor out = input.unfold2d(options);

        CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults())
                .execute(ExecutionMode.FORWARD);

        assertArrayEquals(new int[]{1, 4, 9}, out.getShape());
        assertArrayEquals(new double[]{
                0, 0, 0, 0, 1, 2, 0, 3, 4,
                0, 0, 0, 1, 2, 0, 3, 4, 0,
                0, 1, 2, 0, 3, 4, 0, 0, 0,
                1, 2, 0, 3, 4, 0, 0, 0, 0
        }, out.toDoubleArrayCopy(), 1e-6);
    }

    @Test
    void fold2dAccumulatesOverlappingColumns() {
        Tensor input = new Tensor(new double[]{
                1, 2, 3,
                4, 5, 6,
                7, 8, 9
        }, new int[]{1, 1, 3, 3}, null, "input", DataType.FLOAT64);
        Tensor out = input.unfold2d(Window2dOptions.of(2, 2))
                .fold2d(new int[]{1, 1, 3, 3}, Window2dOptions.of(2, 2));

        CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults())
                .execute(ExecutionMode.FORWARD);

        assertArrayEquals(new int[]{1, 1, 3, 3}, out.getShape());
        assertArrayEquals(new double[]{
                1, 4, 3,
                8, 20, 12,
                7, 16, 9
        }, out.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void unfold2dBackwardUsesFold2dAccumulation() {
        Tensor input = new Tensor(new double[]{
                1, 2, 3,
                4, 5, 6,
                7, 8, 9
        }, new int[]{1, 1, 3, 3}, null, "input", DataType.FLOAT64);
        input.setRequiresGrad(true);
        Tensor loss = input.unfold2d(Window2dOptions.of(2, 2)).sum();

        CompiledGraph.compile(loss, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.trainingDefaults())
                .execute(ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(new double[]{
                1, 2, 1,
                2, 4, 2,
                1, 2, 1
        }, input.getGradient().toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void unfold2dRejectsInvalidInputs() {
        Tensor rank3 = new Tensor(new double[]{1, 2, 3, 4}, new int[]{1, 2, 2}, null, "rank3", DataType.FLOAT64);
        Tensor ints = new Tensor(new int[]{1, 2, 3, 4}, new int[]{1, 1, 2, 2}, null, "ints", DataType.INT32);

        assertThrows(IllegalArgumentException.class, () -> rank3.unfold2d(Window2dOptions.of(2, 2)));
        assertThrows(IllegalArgumentException.class, () -> ints.unfold2d(Window2dOptions.of(2, 2)));
        assertThrows(IllegalArgumentException.class, () -> new Window2dOptions(0, 2, 1, 1, 0, 0, 1, 1));
    }

    @Test
    void fold2dRejectsIncompatibleShapes() {
        Tensor columns = new Tensor(new double[]{1, 2, 3, 4}, new int[]{1, 4, 1}, null, "columns", DataType.FLOAT64);

        assertThrows(IllegalArgumentException.class,
                () -> columns.fold2d(new int[]{1, 2, 2, 2}, Window2dOptions.of(2, 2)));
        assertThrows(IllegalArgumentException.class,
                () -> columns.fold2d(new int[]{1, 1, 4}, Window2dOptions.of(2, 2)));
    }
}
