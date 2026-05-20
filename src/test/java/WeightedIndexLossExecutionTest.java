import backend.runtime.ExecutionMode;
import config.compile.CompileConfig;
import config.runtime.RuntimeConfig;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.loss.LossReduction;
import tensor.Tensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

public class WeightedIndexLossExecutionTest {

    @Test
    void weightedNllLossFromIndicesReductionModesMatchExpectedValues() {
        Tensor logits = new Tensor(new double[]{
                1.0, 2.0, 3.0,
                0.0, 0.0, 0.0
        }, new int[]{2, 3}, null, "logits", DataType.FLOAT64);
        Tensor logProbs = logits.logSoftmax(1);
        Tensor targetIndices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "targetIndices", DataType.INT32);
        Tensor classWeights = new Tensor(new double[]{0.5, 1.0, 2.0}, new int[]{3}, null, "classWeights", DataType.FLOAT64);

        Tensor none = logProbs.nllLossFromIndices(targetIndices, 1, classWeights, LossReduction.NONE);
        Tensor sum = logProbs.nllLossFromIndices(targetIndices, 1, classWeights, LossReduction.SUM);
        Tensor mean = logProbs.nllLossFromIndices(targetIndices, 1, classWeights, LossReduction.MEAN);

        graph.CompiledGraph.compile(none, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);
        graph.CompiledGraph.compile(sum, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);
        graph.CompiledGraph.compile(mean, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        double[] expectedNone = new double[]{
                0.4076059644443804 * 2.0,
                1.0986122886681098 * 0.5
        };
        assertArrayEquals(expectedNone, none.toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{expectedNone[0] + expectedNone[1]}, sum.toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{(expectedNone[0] + expectedNone[1]) / (2.0 + 0.5)}, mean.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void weightedCrossEntropyWithIgnoreIndexUsesWeightedMeanOverValidSamples() {
        Tensor logits = new Tensor(new double[]{
                1.0, 2.0, 3.0,
                0.0, 0.0, 0.0
        }, new int[]{2, 3}, null, "logits", DataType.FLOAT64);
        Tensor targetIndices = new Tensor(new int[]{2, -1}, new int[]{2}, null, "targetIndices", DataType.INT32);
        Tensor classWeights = new Tensor(new double[]{0.5, 1.0, 2.0}, new int[]{3}, null, "classWeights", DataType.FLOAT64);

        Tensor none = logits.crossEntropyLossFromIndices(targetIndices, 1, -1, classWeights, LossReduction.NONE);
        Tensor sum = logits.crossEntropyLossFromIndices(targetIndices, 1, -1, classWeights, LossReduction.SUM);
        Tensor mean = logits.crossEntropyLossFromIndices(targetIndices, 1, -1, classWeights, LossReduction.MEAN);

        graph.CompiledGraph.compile(none, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);
        graph.CompiledGraph.compile(sum, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);
        graph.CompiledGraph.compile(mean, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        double weighted = 0.4076059644443804 * 2.0;
        assertArrayEquals(new double[]{weighted, 0.0}, none.toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{weighted}, sum.toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{weighted / 2.0}, mean.toDoubleArrayCopy(), 1e-9);
    }
}
