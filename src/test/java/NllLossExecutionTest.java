import runtime.contract.ExecutionMode;
import config.compile.CompileConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class NllLossExecutionTest {
    @Test
    void nllLossMatchesReferenceForOneHotTargets() {
        Tensor logits = new Tensor(new double[]{
                1.0, 2.0, 3.0,
                0.0, 0.0, 0.0
        }, new int[]{2, 3}, null, "logits", DataType.FLOAT64);
        Tensor logProbs = logits.logSoftmax(1);
        Tensor targets = new Tensor(new double[]{
                0.0, 0.0, 1.0,
                1.0, 0.0, 0.0
        }, new int[]{2, 3}, null, "targets", DataType.FLOAT64);

        Tensor loss = logProbs.nllLoss(targets, 1);
        CompiledGraph compiledGraph = CompiledGraph.compile(loss, CompileConfig.noGraphOptimizationBaseline());
        compiledGraph.prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        double[] row0 = logSoftmaxRow(new double[]{1.0, 2.0, 3.0});
        double[] row1 = logSoftmaxRow(new double[]{0.0, 0.0, 0.0});
        double expected = (-(row0[2]) - (row1[0])) / 2.0;
        assertArrayEquals(new double[]{expected}, loss.toDoubleArrayCopy(), 1e-9);
        assertTrue(containsOp(compiledGraph, Operation.OpType.NLL_LOSS));
    }

    @Test
    void nllLossBackwardMatchesReferenceForOneHotTargets() {
        double[] row0 = logSoftmaxRow(new double[]{1.0, 2.0, 3.0});
        double[] row1 = logSoftmaxRow(new double[]{0.0, 0.0, 0.0});
        Tensor logProbs = new Tensor(new double[]{
                row0[0], row0[1], row0[2],
                row1[0], row1[1], row1[2]
        }, new int[]{2, 3}, null, "logProbs", DataType.FLOAT64);
        logProbs.setRequiresGrad(true);
        Tensor targets = new Tensor(new double[]{
                0.0, 0.0, 1.0,
                1.0, 0.0, 0.0
        }, new int[]{2, 3}, null, "targets", DataType.FLOAT64);

        Tensor loss = logProbs.nllLoss(targets, 1);
        CompiledGraph.compile(loss, CompileConfig.training())
                .prepare(RuntimeConfig.trainingDefaults()).execute(ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(new double[]{
                0.0, 0.0, -0.5,
                -0.5, 0.0, 0.0
        }, logProbs.getGradient().toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void nllLossSupportsDistributionTargets() {
        Tensor logProbs = new Tensor(new double[]{
                -1.0, -0.5, -2.0
        }, new int[]{1, 3}, null, "logProbs", DataType.FLOAT64);
        Tensor targets = new Tensor(new double[]{
                0.2, 0.3, 0.5
        }, new int[]{1, 3}, null, "targets", DataType.FLOAT64);

        Tensor loss = logProbs.nllLoss(targets, 1);
        CompiledGraph.compile(loss, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        double expected = -(0.2 * -1.0 + 0.3 * -0.5 + 0.5 * -2.0);
        assertArrayEquals(new double[]{expected}, loss.toDoubleArrayCopy(), 1e-9);
    }

    private static boolean containsOp(CompiledGraph compiledGraph, Operation.OpType opType) {
        return compiledGraph.program().compiledNodes().stream()
                .map(graph.model.CompiledNode::operation)
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
}
