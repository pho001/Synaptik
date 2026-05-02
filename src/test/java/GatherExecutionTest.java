import backend.runtime.ExecutionMode;
import config.optimizer.OptimizerConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import operations.Operation;
import operations.index.gatherGrad;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorPrimitiveBuilder;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class GatherExecutionTest {

    @Test
    void gatherAxisOneSelectsOneValuePerRow() {
        Tensor x = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "x", DataType.FLOAT64);
        Tensor indices = new Tensor(new double[]{2, 0}, new int[]{2}, null, "indices", DataType.FLOAT64);
        Tensor y = x.gather(indices, 1);

        CompiledGraph compiledGraph = CompiledGraph.compile(y, OptimizerConfig.noOptimization());
        compiledGraph.execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new int[]{2}, y.getShape());
        assertArrayEquals(new double[]{3.0, 4.0}, y.toDoubleArrayCopy(), 1e-9);
        assertTrue(containsOp(compiledGraph, Operation.OpType.GATHER));
    }

    @Test
    void gatherSupportsNonContiguousInputView() {
        Tensor base = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "base", DataType.FLOAT64);
        Tensor view = base.permute(1, 0);
        Tensor indices = new Tensor(new double[]{2, 0}, new int[]{2}, null, "indices", DataType.FLOAT64);
        Tensor y = view.gather(indices, 0);

        CompiledGraph.compile(y, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{3.0, 4.0}, y.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void gatherBackwardScattersGradientToSelectedPositions() {
        Tensor x = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "x", DataType.FLOAT64);
        x.setRequiresGrad(true);
        Tensor indices = new Tensor(new double[]{2, 0}, new int[]{2}, null, "indices", DataType.FLOAT64);
        Tensor y = x.gather(indices, 1);

        CompiledGraph.compile(y, OptimizerConfig.trainingDefaults())
                .execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(new double[]{
                0.0, 0.0, 1.0,
                1.0, 0.0, 0.0
        }, x.getGradient().toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void gatherGradPrimitiveScattersIntoOriginalInputShape() {
        Tensor indices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "indices", DataType.INT32);
        Tensor outGrad = new Tensor(new double[]{1, 2}, new int[]{2}, null, "outGrad", DataType.FLOAT64);
        Tensor grad = TensorPrimitiveBuilder.binary(
                indices,
                outGrad,
                new int[]{2, 3},
                new gatherGrad(1),
                "gatherGrad",
                DataType.FLOAT64
        );

        CompiledGraph.compile(grad, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{
                0.0, 0.0, 1.0,
                2.0, 0.0, 0.0
        }, grad.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void gatherRejectsNonIntegralIndices() {
        Tensor x = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "x", DataType.FLOAT64);
        Tensor indices = new Tensor(new double[]{1.5, 0.0}, new int[]{2}, null, "indices", DataType.FLOAT64);
        Tensor y = x.gather(indices, 1);

        assertThrows(IllegalArgumentException.class, () ->
                CompiledGraph.compile(y, OptimizerConfig.noOptimization())
                        .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD));
    }

    private static boolean containsOp(CompiledGraph compiledGraph, Operation.OpType opType) {
        return compiledGraph.getCompiledGraphAsList().stream()
                .map(Tensor::getOperation)
                .filter(op -> op != null)
                .map(Operation::opType)
                .anyMatch(type -> type == opType);
    }
}
