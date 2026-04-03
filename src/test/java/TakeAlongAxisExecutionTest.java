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

public class TakeAlongAxisExecutionTest {

    @Test
    void takeAlongAxisPreservesRankAndUsesIndicesShape() {
        Tensor x = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "x", DataType.FLOAT64);
        Tensor indices = new Tensor(new int[]{2, 1, 0, 0}, new int[]{2, 2}, null, "indices", DataType.INT32);
        Tensor y = x.takeAlongAxis(indices, 1);

        CompiledGraph compiledGraph = CompiledGraph.compile(y, OptimizerConfig.noOptimization());
        compiledGraph.execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new int[]{2, 2}, y.getShape());
        assertArrayEquals(new double[]{3.0, 2.0, 4.0, 4.0}, y.toDoubleArrayCopy(), 1e-9);
        assertTrue(containsOp(compiledGraph, Operation.OpType.TAKE_ALONG_AXIS));
    }

    @Test
    void takeAlongAxisBackwardScattersGradientToSelectedPositions() {
        Tensor x = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "x", DataType.FLOAT64);
        x.setRequiresGrad(true);
        Tensor indices = new Tensor(new int[]{2, 1, 0, 0}, new int[]{2, 2}, null, "indices", DataType.INT32);
        Tensor y = x.takeAlongAxis(indices, 1);

        CompiledGraph.compile(y, OptimizerConfig.trainingDefaults())
                .execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(new double[]{
                0.0, 1.0, 1.0,
                2.0, 0.0, 0.0
        }, x.getGradient().toDoubleArrayCopy(), 1e-9);
    }

    private static boolean containsOp(CompiledGraph compiledGraph, Operation.OpType opType) {
        return compiledGraph.getCompiledGraphAsList().stream()
                .map(Tensor::getOperation)
                .filter(op -> op != null)
                .map(Operation::opType)
                .anyMatch(type -> type == opType);
    }
}
