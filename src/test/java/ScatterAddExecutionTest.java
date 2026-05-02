import backend.runtime.ExecutionMode;
import config.optimizer.OptimizerConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class ScatterAddExecutionTest {

    @Test
    void scatterAddAxisOneAddsIntoSelectedPositions() {
        Tensor base = new Tensor(new double[]{10, 20, 30, 40, 50, 60}, new int[]{2, 3}, null, "base", DataType.FLOAT64);
        Tensor indices = new Tensor(new double[]{2, 0}, new int[]{2}, null, "indices", DataType.FLOAT64);
        Tensor src = new Tensor(new double[]{1, 5}, new int[]{2}, null, "src", DataType.FLOAT64);
        Tensor out = base.scatterAdd(indices, src, 1);

        CompiledGraph compiledGraph = CompiledGraph.compile(out, OptimizerConfig.noOptimization());
        compiledGraph.execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{10, 20, 31, 45, 50, 60}, out.toDoubleArrayCopy(), 1e-9);
        assertTrue(containsOp(compiledGraph, Operation.OpType.SCATTER_ADD));
    }

    @Test
    void scatterAddRepeatedAxisIndicesAccumulateIntoCpuContractPositions() {
        Tensor base = new Tensor(new double[]{
                10, 20, 30,
                40, 50, 60
        }, new int[]{2, 3}, null, "base", DataType.FLOAT64);
        Tensor indices = new Tensor(new int[]{0, 0, 1}, new int[]{3}, null, "indices", DataType.INT32);
        Tensor src = new Tensor(new double[]{1, 2, 3}, new int[]{3}, null, "src", DataType.FLOAT64);
        Tensor out = base.scatterAdd(indices, src, 0);

        CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{
                11, 22, 30,
                40, 50, 63
        }, out.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void scatterAddRepeatedAxisValuesRemainLaneScoped() {
        Tensor base = new Tensor(new double[]{
                10, 20, 30,
                40, 50, 60
        }, new int[]{2, 3}, null, "base", DataType.FLOAT64);
        Tensor indices = new Tensor(new int[]{1, 1}, new int[]{2}, null, "indices", DataType.INT32);
        Tensor src = new Tensor(new double[]{5, 7}, new int[]{2}, null, "src", DataType.FLOAT64);
        Tensor out = base.scatterAdd(indices, src, 1);

        CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{
                10, 25, 30,
                40, 57, 60
        }, out.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void scatterAddRejectsOutOfBoundsIndexAtExecution() {
        Tensor base = new Tensor(new double[]{
                10, 20, 30,
                40, 50, 60
        }, new int[]{2, 3}, null, "base", DataType.FLOAT64);
        Tensor indices = new Tensor(new int[]{3, 0}, new int[]{2}, null, "indices", DataType.INT32);
        Tensor src = new Tensor(new double[]{1, 5}, new int[]{2}, null, "src", DataType.FLOAT64);
        Tensor out = base.scatterAdd(indices, src, 1);

        assertThrows(IllegalArgumentException.class, () ->
                CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                        .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD));
    }

    @Test
    void scatterAddBackwardPropagatesToBaseAndGatherToSource() {
        Tensor base = new Tensor(new double[]{10, 20, 30, 40, 50, 60}, new int[]{2, 3}, null, "base", DataType.FLOAT64);
        base.setRequiresGrad(true);
        Tensor indices = new Tensor(new double[]{2, 0}, new int[]{2}, null, "indices", DataType.FLOAT64);
        Tensor src = new Tensor(new double[]{1, 5}, new int[]{2}, null, "src", DataType.FLOAT64);
        src.setRequiresGrad(true);
        Tensor out = base.scatterAdd(indices, src, 1);

        CompiledGraph.compile(out, OptimizerConfig.trainingDefaults())
                .execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(new double[]{1, 1, 1, 1, 1, 1}, base.getGradient().toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{1, 1}, src.getGradient().toDoubleArrayCopy(), 1e-9);
    }

    private static boolean containsOp(CompiledGraph compiledGraph, Operation.OpType opType) {
        return compiledGraph.getCompiledGraphAsList().stream()
                .map(Tensor::getOperation)
                .filter(op -> op != null)
                .map(Operation::opType)
                .anyMatch(type -> type == opType);
    }
}
