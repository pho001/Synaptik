import backend.runtime.ExecutionMode;
import config.compile.CompileConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

public class ReductionBroadcastContractTest {
    @Test
    void keepDimsPreservesReducedAxisAcrossNumericAndBoolReductions() {
        Tensor x = new Tensor(new double[]{
                1, 2, 3,
                4, 5, 6
        }, new int[]{2, 3}, null, "x", DataType.FLOAT64);
        Tensor mask = new Tensor(new byte[]{
                1, 1, 0,
                1, 0, 0
        }, new int[]{2, 3}, null, "mask", DataType.BOOL);

        Tensor sum = x.sum(1, true);
        Tensor mean = x.mean(1, true);
        Tensor min = x.min(1, true);
        Tensor max = x.max(1, true);
        Tensor all = mask.all(1, true);
        Tensor any = mask.any(1, true);

        CompiledGraph.compile(sum, CompileConfig.noGraphOptimizationBaseline()).prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);
        CompiledGraph.compile(mean, CompileConfig.noGraphOptimizationBaseline()).prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);
        CompiledGraph.compile(min, CompileConfig.noGraphOptimizationBaseline()).prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);
        CompiledGraph.compile(max, CompileConfig.noGraphOptimizationBaseline()).prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);
        CompiledGraph.compile(all, CompileConfig.noGraphOptimizationBaseline()).prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);
        CompiledGraph.compile(any, CompileConfig.noGraphOptimizationBaseline()).prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        assertArrayEquals(new int[]{2, 1}, sum.getShape());
        assertArrayEquals(new int[]{2, 1}, mean.getShape());
        assertArrayEquals(new int[]{2, 1}, min.getShape());
        assertArrayEquals(new int[]{2, 1}, max.getShape());
        assertArrayEquals(new int[]{2, 1}, all.getShape());
        assertArrayEquals(new int[]{2, 1}, any.getShape());

        assertArrayEquals(new double[]{6.0, 15.0}, sum.toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{2.0, 5.0}, mean.toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{1.0, 4.0}, min.toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{3.0, 6.0}, max.toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new boolean[]{false, false}, all.toBooleanArrayCopy());
        assertArrayEquals(new boolean[]{true, true}, any.toBooleanArrayCopy());
    }

    @Test
    void keepDimsReductionCanFeedNaturalBroadcastChainWithoutExplicitReshape() {
        Tensor x = new Tensor(new double[]{
                1, 2, 3,
                4, 5, 6
        }, new int[]{2, 3}, null, "x", DataType.FLOAT64);

        Tensor centeredBySum = x.sub(x.sum(1, true).mul(1.0 / 3.0));
        Tensor centeredByMean = x.sub(x.mean(1, true));

        CompiledGraph.compile(centeredBySum, CompileConfig.noGraphOptimizationBaseline()).prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);
        CompiledGraph.compile(centeredByMean, CompileConfig.noGraphOptimizationBaseline()).prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        assertArrayEquals(new int[]{2, 3}, centeredBySum.getShape());
        assertArrayEquals(new int[]{2, 3}, centeredByMean.getShape());
        assertArrayEquals(new double[]{-1, 0, 1, -1, 0, 1}, centeredBySum.toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{-1, 0, 1, -1, 0, 1}, centeredByMean.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void backwardThroughKeepDimsMeanBroadcastChainUsesReductionContractCorrectly() {
        Tensor x = new Tensor(new double[]{
                1, 2, 3,
                4, 5, 6
        }, new int[]{2, 3}, null, "x", DataType.FLOAT64);
        x.setRequiresGrad(true);

        Tensor centered = x.sub(x.mean(1, true));
        Tensor loss = centered.mul(centered).sum();

        CompiledGraph.compile(loss, CompileConfig.training())
                .prepare(RuntimeConfig.trainingDefaults()).execute(ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(new int[]{2, 3}, x.getGradient().getShape());
        assertArrayEquals(new double[]{-2.0, 0.0, 2.0, -2.0, 0.0, 2.0}, x.getGradient().toDoubleArrayCopy(), 1e-9);
    }
}
