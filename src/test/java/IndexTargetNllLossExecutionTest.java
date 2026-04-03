import backend.runtime.ExecutionMode;
import config.optimizer.OptimizerConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

public class IndexTargetNllLossExecutionTest {

    @Test
    void nllLossFromIndicesMatchesDenseOneHotNllLoss() {
        Tensor logitsA = new Tensor(new double[]{
                1.0, 2.0, 3.0,
                0.0, 0.0, 0.0
        }, new int[]{2, 3}, null, "logitsA", DataType.FLOAT64);
        Tensor logProbsA = logitsA.logSoftmax(1);
        Tensor oneHotTargets = new Tensor(new double[]{
                0.0, 0.0, 1.0,
                1.0, 0.0, 0.0
        }, new int[]{2, 3}, null, "oneHotTargets", DataType.FLOAT64);
        Tensor denseLoss = logProbsA.nllLoss(oneHotTargets, 1);
        CompiledGraph.compile(denseLoss, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor logitsB = new Tensor(new double[]{
                1.0, 2.0, 3.0,
                0.0, 0.0, 0.0
        }, new int[]{2, 3}, null, "logitsB", DataType.FLOAT64);
        Tensor logProbsB = logitsB.logSoftmax(1);
        Tensor targetIndices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "targetIndices", DataType.INT32);
        Tensor indexLoss = logProbsB.nllLossFromIndices(targetIndices, 1);
        CompiledGraph.compile(indexLoss, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(denseLoss.toDoubleArrayCopy(), indexLoss.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void nllLossFromIndicesBackwardMatchesReference() {
        double[] row0 = logSoftmaxRow(new double[]{1.0, 2.0, 3.0});
        double[] row1 = logSoftmaxRow(new double[]{0.0, 0.0, 0.0});
        Tensor logProbs = new Tensor(new double[]{
                row0[0], row0[1], row0[2],
                row1[0], row1[1], row1[2]
        }, new int[]{2, 3}, null, "logProbs", DataType.FLOAT64);
        logProbs.setRequiresGrad(true);
        Tensor targetIndices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "targetIndices", DataType.INT32);
        Tensor loss = logProbs.nllLossFromIndices(targetIndices, 1);

        CompiledGraph.compile(loss, OptimizerConfig.trainingDefaults())
                .execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(new double[]{
                0.0, 0.0, -0.5,
                -0.5, 0.0, 0.0
        }, logProbs.getGradient().toDoubleArrayCopy(), 1e-9);
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
