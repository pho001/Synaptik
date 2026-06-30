import backend.runtime.ExecutionMode;
import config.compile.CompileConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class OnnxWave3CoreOpsExecutionTest {
    @Test
    void cumSumScansForwardExclusiveAndReverse() {
        Tensor x = new Tensor(new double[]{
                1, 2, 3,
                4, 5, 6
        }, new int[]{2, 3}, null, "x", DataType.FLOAT64);

        Tensor forward = x.cumSum(1);
        CompiledGraph forwardGraph = CompiledGraph.compile(forward, CompileConfig.noGraphOptimizationBaseline());
        forwardGraph.prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{1, 3, 6, 4, 9, 15}, forward.toDoubleArrayCopy(), 1e-9);
        assertTrue(containsOp(forwardGraph, Operation.OpType.CUMSUM));

        Tensor exclusiveReverse = x.cumSum(1, true, true);
        CompiledGraph.compile(exclusiveReverse, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{5, 3, 0, 11, 6, 0}, exclusiveReverse.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void cumSumSupportsInt32AndNonContiguousInput() {
        Tensor x = new Tensor(new int[]{
                1, 2, 3,
                4, 5, 6
        }, new int[]{2, 3}, null, "x", DataType.INT32);

        Tensor transposed = x.transpose();
        Tensor out = transposed.cumSum(1);
        CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        assertArrayEquals(new int[]{3, 2}, out.getShape());
        assertArrayEquals(new double[]{1, 5, 2, 7, 3, 9}, out.toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new int[]{1, 5, 2, 7, 3, 9}, out.toInt32ArrayCopy());
    }

    @Test
    void cumSumRejectsBoolInput() {
        Tensor mask = new Tensor(new byte[]{1, 0, 1}, new int[]{3}, null, "mask", DataType.BOOL);

        assertThrows(IllegalArgumentException.class, () -> mask.cumSum(0));
    }

    private static boolean containsOp(CompiledGraph compiledGraph, Operation.OpType opType) {
        return compiledGraph.program().compiledNodes().stream()
                .map(graph.model.CompiledNode::operation)
                .filter(op -> op != null)
                .map(Operation::opType)
                .anyMatch(type -> type == opType);
    }
}
