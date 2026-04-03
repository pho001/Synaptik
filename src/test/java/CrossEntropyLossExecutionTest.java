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

public class CrossEntropyLossExecutionTest {
    @Test
    void crossEntropyLossMatchesLogSoftmaxPlusNllLoss() {
        Tensor logitsA = new Tensor(new double[]{
                1.0, 2.0, 3.0,
                0.0, 0.0, 0.0
        }, new int[]{2, 3}, null, "logitsA", DataType.FLOAT64);
        logitsA.setRequiresGrad(true);
        Tensor targetsA = new Tensor(new double[]{
                0.0, 0.0, 1.0,
                1.0, 0.0, 0.0
        }, new int[]{2, 3}, null, "targetsA", DataType.FLOAT64);

        Tensor reference = logitsA.logSoftmax(1).nllLoss(targetsA, 1);
        CompiledGraph.compile(reference, OptimizerConfig.trainingDefaults())
                .execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);
        double[] referenceLoss = reference.toDoubleArrayCopy();
        double[] referenceGrad = logitsA.getGradient().toDoubleArrayCopy();

        Tensor logitsB = new Tensor(new double[]{
                1.0, 2.0, 3.0,
                0.0, 0.0, 0.0
        }, new int[]{2, 3}, null, "logitsB", DataType.FLOAT64);
        logitsB.setRequiresGrad(true);
        Tensor targetsB = new Tensor(new double[]{
                0.0, 0.0, 1.0,
                1.0, 0.0, 0.0
        }, new int[]{2, 3}, null, "targetsB", DataType.FLOAT64);

        Tensor direct = logitsB.crossEntropyLoss(targetsB, 1);
        CompiledGraph compiledGraph = CompiledGraph.compile(direct, OptimizerConfig.trainingDefaults());
        compiledGraph
                .execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(referenceLoss, direct.toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(referenceGrad, logitsB.getGradient().toDoubleArrayCopy(), 1e-9);
        assertTrue(compiledGraph.getCompiledGraphAsList().stream()
                .map(Tensor::getOperation)
                .filter(op -> op != null)
                .map(Operation::opType)
                .anyMatch(opType -> opType == Operation.OpType.CROSS_ENTROPY_LOSS));
    }
}
