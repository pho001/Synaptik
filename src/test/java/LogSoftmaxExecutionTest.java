import backend.runtime.ExecutionMode;
import config.optimizer.OptimizerConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class LogSoftmaxExecutionTest {
    @Test
    void logSoftmaxAxisMatchesReference() {
        Tensor logits = new Tensor(new double[]{
                1.0, 2.0, 3.0,
                0.0, 0.0, 0.0
        }, new int[]{2, 3}, null, "logits", DataType.FLOAT64);

        Tensor logProbs = logits.logSoftmax(1);
        CompiledGraph compiledGraph = CompiledGraph.compile(logProbs, OptimizerConfig.noOptimization());
        compiledGraph.execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        double[] actual = logProbs.toDoubleArrayCopy();
        assertArrayEquals(logSoftmaxRow(new double[]{1.0, 2.0, 3.0}), new double[]{actual[0], actual[1], actual[2]}, 1e-9);
        assertArrayEquals(logSoftmaxRow(new double[]{0.0, 0.0, 0.0}), new double[]{actual[3], actual[4], actual[5]}, 1e-9);
        assertTrue(containsOp(compiledGraph, Operation.OpType.LOG_SOFTMAX));
    }

    @Test
    void logSoftmaxIsNumericallyStableForLargeLogits() {
        Tensor logits = new Tensor(new double[]{1000.0, 1001.0, 1002.0}, new int[]{1, 3}, null, "logits", DataType.FLOAT64);

        Tensor logProbs = logits.logSoftmax(1);
        CompiledGraph.compile(logProbs, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(logSoftmaxRow(new double[]{1000.0, 1001.0, 1002.0}), logProbs.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void expOfLogSoftmaxMatchesSoftmax() {
        Tensor logits = new Tensor(new double[]{1.0, 2.0, 3.0}, new int[]{1, 3}, null, "logits", DataType.FLOAT64);

        Tensor softmax = logits.softmax(1);
        Tensor logSoftmax = logits.logSoftmax(1);
        Tensor reconstructed = logSoftmax.exp();

        CompiledGraph.compile(softmax, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
        CompiledGraph.compile(reconstructed, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(softmax.toDoubleArrayCopy(), reconstructed.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void logSoftmaxBackwardMatchesReferenceForSeededOnes() {
        Tensor logits = new Tensor(new double[]{1.0, 2.0, 3.0}, new int[]{1, 3}, null, "logits", DataType.FLOAT64);
        logits.setRequiresGrad(true);
        Tensor logProbs = logits.logSoftmax(1);

        CompiledGraph.compile(logProbs, OptimizerConfig.trainingDefaults())
                .execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);

        double[] softmax = softmaxRow(new double[]{1.0, 2.0, 3.0});
        assertArrayEquals(new double[]{
                1.0 - 3.0 * softmax[0],
                1.0 - 3.0 * softmax[1],
                1.0 - 3.0 * softmax[2]
        }, logits.getGradient().toDoubleArrayCopy(), 1e-9);
    }

    private static boolean containsOp(CompiledGraph compiledGraph, Operation.OpType opType) {
        return compiledGraph.getCompiledGraphAsList().stream()
                .map(Tensor::getOperation)
                .filter(op -> op != null)
                .map(Operation::opType)
                .anyMatch(type -> type == opType);
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

    private static double[] softmaxRow(double[] values) {
        double[] log = logSoftmaxRow(values);
        double[] out = new double[log.length];
        for (int i = 0; i < log.length; i++) {
            out[i] = Math.exp(log[i]);
        }
        return out;
    }
}
