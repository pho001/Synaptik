import backend.runtime.ExecutionMode;
import config.compile.CompileConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import operations.index.ScatterReduction;
import operations.reduction.ArgMaxTiePolicy;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.options.Window2dOptions;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

public class WindowDagPrimitiveTest {
    @Test
    void maxPool2dForwardCanBeExpressedWithUnfoldAndReduceMax() {
        Tensor input = new Tensor(new double[]{
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16
        }, new int[]{1, 1, 4, 4}, null, "input", DataType.FLOAT64);
        Window2dOptions options = new Window2dOptions(2, 2, 2, 2, 0, 0, 1, 1);

        Tensor out = input.unfold2d(options)
                .reshape(1, 1, 4, 4)
                .max(2)
                .reshape(1, 1, 2, 2);
        CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults())
                .execute(ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{
                6, 8,
                14, 16
        }, out.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void maxPool2dBackwardTiePathCanBeExpressedWithArgMaxScatterAndFold() {
        Tensor inputGrad = maxPool2dBackwardTiePath(ArgMaxTiePolicy.FIRST_INDEX);

        assertArrayEquals(new double[]{
                1, 0,
                0, 0
        }, inputGrad.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void maxPool2dBackwardLastIndexTiePathCanBeExpressedWithArgMaxScatterAndFold() {
        Tensor inputGrad = maxPool2dBackwardTiePath(ArgMaxTiePolicy.LAST_INDEX);

        assertArrayEquals(new double[]{
                0, 0,
                0, 1
        }, inputGrad.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void scatterSelectsWindowPositionsAndFoldAccumulatesSpatialOverlap() {
        Tensor columns = new Tensor(new double[16], new int[]{1, 4, 4}, null, "columns", DataType.FLOAT64);
        Tensor indices = new Tensor(new long[]{
                1, 0, 0, 0
        }, new int[]{1, 1, 4}, null, "indices", DataType.INT64);
        Tensor updates = new Tensor(new double[]{
                2, 3, 0, 0
        }, new int[]{1, 1, 4}, null, "updates", DataType.FLOAT64);
        Tensor columnGrad = columns.scatterElements(indices, updates, 1, ScatterReduction.NONE);
        Tensor inputGrad = columnGrad.fold2d(new int[]{1, 1, 3, 3}, Window2dOptions.of(2, 2));

        CompiledGraph.compile(columnGrad, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults())
                .execute(ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{
                0, 3, 0, 0,
                2, 0, 0, 0,
                0, 0, 0, 0,
                0, 0, 0, 0
        }, columnGrad.toDoubleArrayCopy(), 1e-9);

        CompiledGraph.compile(inputGrad, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults())
                .execute(ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{
                0, 5, 0,
                0, 0, 0,
                0, 0, 0
        }, inputGrad.toDoubleArrayCopy(), 1e-9);
    }

    private static Tensor maxPool2dBackwardTiePath(ArgMaxTiePolicy tiePolicy) {
        Tensor input = new Tensor(new double[]{
                5, 5,
                5, 5
        }, new int[]{1, 1, 2, 2}, null, "input", DataType.FLOAT64);
        Window2dOptions options = Window2dOptions.of(2, 2);

        Tensor columns = input.unfold2d(options).reshape(1, 1, 4, 1);
        Tensor winner = columns.argMax(2, true, tiePolicy);
        Tensor updates = Tensor.onesLike(columns.max(2, true));
        Tensor columnGrad = Tensor.zerosLike(columns)
                .scatterElements(winner, updates, 2, ScatterReduction.NONE);
        Tensor inputGrad = columnGrad.reshape(1, 4, 1)
                .fold2d(new int[]{1, 1, 2, 2}, options);

        CompiledGraph.compile(inputGrad, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults())
                .execute(ExecutionMode.FORWARD);

        return inputGrad;
    }
}
