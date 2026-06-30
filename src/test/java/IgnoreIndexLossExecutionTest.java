import runtime.contract.ExecutionMode;
import config.compile.CompileConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

public class IgnoreIndexLossExecutionTest {

    @Test
    void nllLossFromIndicesIgnoreIndexSkipsIgnoredSamples() {
        Tensor logits = new Tensor(new double[]{
                1.0, 2.0, 3.0,
                0.0, 0.0, 0.0
        }, new int[]{2, 3}, null, "logits", DataType.FLOAT64);
        Tensor logProbs = logits.logSoftmax(1);
        Tensor targetIndices = new Tensor(new int[]{2, -1}, new int[]{2}, null, "targetIndices", DataType.INT32);

        Tensor loss = logProbs.nllLossFromIndices(targetIndices, 1, -1);
        CompiledGraph.compile(loss, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        double[] row0 = logSoftmaxRow(new double[]{1.0, 2.0, 3.0});
        assertArrayEquals(new double[]{-row0[2]}, loss.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void nllLossFromIndicesIgnoreIndexZerosGradientForIgnoredSamples() {
        double[] row0 = logSoftmaxRow(new double[]{1.0, 2.0, 3.0});
        double[] row1 = logSoftmaxRow(new double[]{0.0, 0.0, 0.0});
        Tensor logProbs = new Tensor(new double[]{
                row0[0], row0[1], row0[2],
                row1[0], row1[1], row1[2]
        }, new int[]{2, 3}, null, "logProbs", DataType.FLOAT64);
        logProbs.setRequiresGrad(true);
        Tensor targetIndices = new Tensor(new int[]{2, -1}, new int[]{2}, null, "targetIndices", DataType.INT32);
        Tensor loss = logProbs.nllLossFromIndices(targetIndices, 1, -1);

        CompiledGraph.compile(loss, CompileConfig.training())
                .prepare(RuntimeConfig.trainingDefaults()).execute(ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(new double[]{
                0.0, 0.0, -1.0,
                0.0, 0.0, 0.0
        }, logProbs.getGradient().toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void crossEntropyLossFromIndicesIgnoreIndexMatchesReference() {
        Tensor logitsA = new Tensor(new double[]{
                1.0, 2.0, 3.0,
                0.0, 0.0, 0.0
        }, new int[]{2, 3}, null, "logitsA", DataType.FLOAT64);
        logitsA.setRequiresGrad(true);
        Tensor targetIndicesA = new Tensor(new int[]{2, -1}, new int[]{2}, null, "targetIndicesA", DataType.INT32);
        Tensor reference = logitsA.logSoftmax(1).nllLossFromIndices(targetIndicesA, 1, -1);
        CompiledGraph.compile(reference, CompileConfig.training())
                .prepare(RuntimeConfig.trainingDefaults()).execute(ExecutionMode.FORWARD_BACKWARD);

        Tensor logitsB = new Tensor(new double[]{
                1.0, 2.0, 3.0,
                0.0, 0.0, 0.0
        }, new int[]{2, 3}, null, "logitsB", DataType.FLOAT64);
        logitsB.setRequiresGrad(true);
        Tensor targetIndicesB = new Tensor(new int[]{2, -1}, new int[]{2}, null, "targetIndicesB", DataType.INT32);
        Tensor direct = logitsB.crossEntropyLossFromIndices(targetIndicesB, 1, -1);
        CompiledGraph.compile(direct, CompileConfig.training())
                .prepare(RuntimeConfig.trainingDefaults()).execute(ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(reference.toDoubleArrayCopy(), direct.toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(logitsA.getGradient().toDoubleArrayCopy(), logitsB.getGradient().toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void allIgnoredSamplesProduceZeroLoss() {
        Tensor logits = new Tensor(new double[]{
                1.0, 2.0, 3.0,
                0.0, 0.0, 0.0
        }, new int[]{2, 3}, null, "logits", DataType.FLOAT64);
        Tensor targetIndices = new Tensor(new int[]{-1, -1}, new int[]{2}, null, "targetIndices", DataType.INT32);
        Tensor loss = logits.crossEntropyLossFromIndices(targetIndices, 1, -1);

        CompiledGraph.compile(loss, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{0.0}, loss.toDoubleArrayCopy(), 1e-9);
    }

    private static double[] logSoftmaxRow(double[] values) {
        double max = Double.NEGATIVE_INFINITY;
        for (double value : values) {
            max = Math.max(max, value);
        }
        double sum = 0.0;
        for (double value : values) {
            sum += Math.exp(value - max);
        }
        double logSumExp = max + Math.log(sum);
        double[] out = new double[values.length];
        for (int i = 0; i < values.length; i++) {
            out[i] = values[i] - logSumExp;
        }
        return out;
    }
}
