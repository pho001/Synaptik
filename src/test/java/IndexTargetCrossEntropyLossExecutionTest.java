import backend.runtime.ExecutionMode;
import config.compile.CompileConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class IndexTargetCrossEntropyLossExecutionTest {

    @Test
    void crossEntropyLossFromIndicesMatchesIndexReference() {
        Tensor logitsA = new Tensor(new double[]{
                1.0, 2.0, 3.0,
                0.0, 0.0, 0.0
        }, new int[]{2, 3}, null, "logitsA", DataType.FLOAT64);
        logitsA.setRequiresGrad(true);
        Tensor targetIndicesA = new Tensor(new int[]{2, 0}, new int[]{2}, null, "targetIndicesA", DataType.INT32);

        Tensor reference = logitsA.logSoftmax(1).nllLossFromIndices(targetIndicesA, 1);
        CompiledGraph.compile(reference, CompileConfig.training())
                .execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);
        double[] referenceLoss = reference.toDoubleArrayCopy();
        double[] referenceGrad = logitsA.getGradient().toDoubleArrayCopy();

        Tensor logitsB = new Tensor(new double[]{
                1.0, 2.0, 3.0,
                0.0, 0.0, 0.0
        }, new int[]{2, 3}, null, "logitsB", DataType.FLOAT64);
        logitsB.setRequiresGrad(true);
        Tensor targetIndicesB = new Tensor(new int[]{2, 0}, new int[]{2}, null, "targetIndicesB", DataType.INT32);

        Tensor direct = logitsB.crossEntropyLossFromIndices(targetIndicesB, 1);
        CompiledGraph compiledGraph = CompiledGraph.compile(direct, CompileConfig.training());
        compiledGraph.execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(referenceLoss, direct.toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(referenceGrad, logitsB.getGradient().toDoubleArrayCopy(), 1e-9);
        assertTrue(compiledGraph.getCompiledGraphAsList().stream()
                .map(Tensor::getOperation)
                .filter(op -> op != null)
                .map(Operation::opType)
                .anyMatch(opType -> opType == Operation.OpType.CROSS_ENTROPY_LOSS_INDICES));
        assertFalse(compiledGraph.getCompiledGraphAsList().stream()
                .map(Tensor::getOperation)
                .filter(op -> op != null)
                .map(Operation::opType)
                .anyMatch(opType -> opType == Operation.OpType.CROSS_ENTROPY_LOSS_INDICES_GRAD));
    }

    @Test
    void crossEntropyLossFromIndicesMatchesDenseOneHotCrossEntropy() {
        Tensor logitsA = new Tensor(new double[]{
                1.0, 2.0, 3.0,
                0.0, 0.0, 0.0
        }, new int[]{2, 3}, null, "logitsA", DataType.FLOAT64);
        Tensor oneHotTargets = new Tensor(new double[]{
                0.0, 0.0, 1.0,
                1.0, 0.0, 0.0
        }, new int[]{2, 3}, null, "oneHotTargets", DataType.FLOAT64);
        Tensor denseLoss = logitsA.crossEntropyLoss(oneHotTargets, 1);
        CompiledGraph.compile(denseLoss, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor logitsB = new Tensor(new double[]{
                1.0, 2.0, 3.0,
                0.0, 0.0, 0.0
        }, new int[]{2, 3}, null, "logitsB", DataType.FLOAT64);
        Tensor targetIndices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "targetIndices", DataType.INT32);
        Tensor indexLoss = logitsB.crossEntropyLossFromIndices(targetIndices, 1);
        CompiledGraph.compile(indexLoss, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(denseLoss.toDoubleArrayCopy(), indexLoss.toDoubleArrayCopy(), 1e-9);
    }
}
