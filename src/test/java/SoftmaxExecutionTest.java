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

public class SoftmaxExecutionTest {
    @Test
    void softmaxAxisProducesPerRowDistribution() {
        Tensor logits = new Tensor(new double[]{
                1.0, 2.0, 3.0,
                0.0, 0.0, 0.0
        }, new int[]{2, 3}, null, "logits", DataType.FLOAT64);

        Tensor probs = logits.softmax(1);
        CompiledGraph compiledGraph = CompiledGraph.compile(probs, OptimizerConfig.noOptimization());
        compiledGraph.execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        double[] actual = probs.toDoubleArrayCopy();
        assertArrayEquals(softmaxRow(new double[]{1.0, 2.0, 3.0}), new double[]{actual[0], actual[1], actual[2]}, 1e-9);
        assertArrayEquals(new double[]{1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0}, new double[]{actual[3], actual[4], actual[5]}, 1e-9);
        assertTrue(containsOp(compiledGraph, Operation.OpType.SOFTMAX));
    }

    @Test
    void softmaxIsNumericallyStableForLargeLogits() {
        Tensor logits = new Tensor(new double[]{1000.0, 1001.0, 1002.0}, new int[]{1, 3}, null, "logits", DataType.FLOAT64);

        Tensor probs = logits.softmax(1);
        CompiledGraph.compile(probs, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(softmaxRow(new double[]{1000.0, 1001.0, 1002.0}), probs.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void softmaxBackwardMatchesReferenceJacobianVectorProduct() {
        Tensor logits = new Tensor(new double[]{1.0, 2.0, 3.0}, new int[]{1, 3}, null, "logits", DataType.FLOAT64);
        logits.setRequiresGrad(true);
        Tensor probs = logits.softmax(1);

        CompiledGraph.compile(probs, OptimizerConfig.trainingDefaults())
                .execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);

        double[] y = probs.toDoubleArrayCopy();
        double sumY = y[0] + y[1] + y[2];
        assertArrayEquals(new double[]{1.0, 1.0, 1.0}, new double[]{sumY, sumY, sumY}, 1e-9);
        assertArrayEquals(new double[]{0.0, 0.0, 0.0}, logits.getGradient().toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void softmaxSupportsFloat32() {
        Tensor logits = new Tensor(new float[]{1.0f, 2.0f, 3.0f}, new int[]{1, 3}, null, "logits", DataType.FLOAT32);
        Tensor probs = logits.softmax(1);

        CompiledGraph.compile(probs, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(softmaxRow(new double[]{1.0, 2.0, 3.0}), probs.toDoubleArrayCopy(), 1e-6);
    }

    private static boolean containsOp(CompiledGraph compiledGraph, Operation.OpType opType) {
        return compiledGraph.getCompiledGraphAsList().stream()
                .map(Tensor::getOperation)
                .filter(op -> op != null)
                .map(Operation::opType)
                .anyMatch(type -> type == opType);
    }

    private static double[] softmaxRow(double[] values) {
        double max = Double.NEGATIVE_INFINITY;
        for (double value : values) {
            max = Math.max(max, value);
        }
        double[] out = new double[values.length];
        double sum = 0.0;
        for (int i = 0; i < values.length; i++) {
            out[i] = Math.exp(values[i] - max);
            sum += out[i];
        }
        for (int i = 0; i < values.length; i++) {
            out[i] /= sum;
        }
        return out;
    }
}
