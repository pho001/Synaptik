import config.optimizer.OptimizerConfig;
import graph.CompiledGraph;
import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class CrossEntropyLossFromIndicesLoweringTest {

    @Test
    void canonicalLogSoftmaxAndIndexNllLowerToSpecializedCrossEntropy() {
        Tensor logits = new Tensor(new double[]{
                1.0, 2.0, 3.0,
                0.0, 0.0, 0.0
        }, new int[]{2, 3}, null, "logits", DataType.FLOAT64);
        Tensor targetIndices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "targetIndices", DataType.INT32);

        Tensor canonical = logits.logSoftmax(1).nllLossFromIndices(targetIndices, 1);
        CompiledGraph compiled = CompiledGraph.compile(canonical, OptimizerConfig.inferenceDefaults());

        assertTrue(compiled.getCompiledGraphAsList().stream()
                .map(Tensor::getOperation)
                .filter(op -> op != null)
                .map(Operation::opType)
                .anyMatch(opType -> opType == Operation.OpType.CROSS_ENTROPY_LOSS_INDICES));

        Tensor directLogits = new Tensor(new double[]{
                1.0, 2.0, 3.0,
                0.0, 0.0, 0.0
        }, new int[]{2, 3}, null, "directLogits", DataType.FLOAT64);
        Tensor direct = directLogits.crossEntropyLossFromIndices(targetIndices, 1);

        graph.CompiledGraph.compile(canonical, OptimizerConfig.inferenceDefaults())
                .execute(config.runtime.RuntimeConfig.inferenceDefaults(), backend.runtime.ExecutionMode.FORWARD);
        graph.CompiledGraph.compile(direct, OptimizerConfig.noOptimization())
                .execute(config.runtime.RuntimeConfig.inferenceDefaults(), backend.runtime.ExecutionMode.FORWARD);

        assertArrayEquals(direct.toDoubleArrayCopy(), canonical.toDoubleArrayCopy(), 1e-9);
    }
}
