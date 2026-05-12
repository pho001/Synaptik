import backend.runtime.ExecutionMode;
import config.compile.CompileConfig;
import config.runtime.RuntimeConfig;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.loss.LossReduction;
import tensor.Tensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

public class IndexLossReductionExecutionTest {

    @Test
    void nllLossFromIndicesReductionModesMatchExpectedShapesAndValues() {
        Tensor logits = new Tensor(new double[]{
                1.0, 2.0, 3.0,
                0.0, 0.0, 0.0
        }, new int[]{2, 3}, null, "logits", DataType.FLOAT64);
        Tensor logProbs = logits.logSoftmax(1);
        Tensor targetIndices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "targetIndices", DataType.INT32);

        Tensor none = logProbs.nllLossFromIndices(targetIndices, 1, LossReduction.NONE);
        Tensor sum = logProbs.nllLossFromIndices(targetIndices, 1, LossReduction.SUM);
        Tensor mean = logProbs.nllLossFromIndices(targetIndices, 1, LossReduction.MEAN);

        graph.CompiledGraph.compile(none, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
        graph.CompiledGraph.compile(sum, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
        graph.CompiledGraph.compile(mean, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        double[] expectedNone = new double[]{0.4076059644443804, 1.0986122886681098};
        assertArrayEquals(new int[]{2}, none.getShape());
        assertArrayEquals(expectedNone, none.toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{expectedNone[0] + expectedNone[1]}, sum.toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{(expectedNone[0] + expectedNone[1]) / 2.0}, mean.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void ignoreIndexWithNoneAndSumBehavesAsExpected() {
        Tensor logits = new Tensor(new double[]{
                1.0, 2.0, 3.0,
                0.0, 0.0, 0.0
        }, new int[]{2, 3}, null, "logits", DataType.FLOAT64);
        Tensor targetIndices = new Tensor(new int[]{2, -1}, new int[]{2}, null, "targetIndices", DataType.INT32);

        Tensor none = logits.crossEntropyLossFromIndices(targetIndices, 1, -1, LossReduction.NONE);
        Tensor sum = logits.crossEntropyLossFromIndices(targetIndices, 1, -1, LossReduction.SUM);

        graph.CompiledGraph.compile(none, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
        graph.CompiledGraph.compile(sum, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new int[]{2}, none.getShape());
        assertArrayEquals(new double[]{0.4076059644443804, 0.0}, none.toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{0.4076059644443804}, sum.toDoubleArrayCopy(), 1e-9);
    }
}
