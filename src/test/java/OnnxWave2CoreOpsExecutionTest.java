import backend.runtime.ExecutionMode;
import config.compile.CompileConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class OnnxWave2CoreOpsExecutionTest {
    @Test
    void reduceProdReducesAxisAndAllElements() {
        Tensor x = new Tensor(new double[]{
                1, 2, 3,
                4, 5, 6
        }, new int[]{2, 3}, null, "x", DataType.FLOAT64);

        Tensor axis = x.prod(1, true);
        CompiledGraph axisGraph = CompiledGraph.compile(axis, CompileConfig.noGraphOptimizationBaseline());
        axisGraph.execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new int[]{2, 1}, axis.getShape());
        assertArrayEquals(new double[]{6.0, 120.0}, axis.toDoubleArrayCopy(), 1e-9);
        assertTrue(containsOp(axisGraph, Operation.OpType.REDUCE_PROD));

        Tensor all = x.prod();
        CompiledGraph.compile(all, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
        assertArrayEquals(new double[]{720.0}, all.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void argMaxReturnsFirstMaxIndexAsInt32() {
        Tensor x = new Tensor(new double[]{
                1, 4, 4,
                7, 6, 7
        }, new int[]{2, 3}, null, "x", DataType.FLOAT64);
        Tensor out = x.argMax(1, false);

        CompiledGraph compiledGraph = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline());
        compiledGraph.execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new int[]{2}, out.getShape());
        assertArrayEquals(new double[]{1.0, 0.0}, out.toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new int[]{1, 0}, out.getInt32Data());
        assertTrue(containsOp(compiledGraph, Operation.OpType.ARGMAX));
    }

    @Test
    void padCopiesInputIntoConstantFrame() {
        Tensor x = new Tensor(new float[]{
                1, 2,
                3, 4
        }, new int[]{2, 2}, null, "x", DataType.FLOAT32);
        Tensor out = x.pad(new int[]{1, 2}, new int[]{0, 1}, -1.0);

        CompiledGraph compiledGraph = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline());
        compiledGraph.execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new int[]{3, 5}, out.getShape());
        assertArrayEquals(new double[]{
                -1, -1, -1, -1, -1,
                -1, -1, 1, 2, -1,
                -1, -1, 3, 4, -1
        }, out.toDoubleArrayCopy(), 1e-6);
        assertTrue(containsOp(compiledGraph, Operation.OpType.PAD));
    }

    @Test
    void tileRepeatsEveryAxis() {
        Tensor x = new Tensor(new double[]{
                1, 2,
                3, 4
        }, new int[]{2, 2}, null, "x", DataType.FLOAT64);
        Tensor out = x.tile(2, 3);

        CompiledGraph compiledGraph = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline());
        compiledGraph.execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new int[]{4, 6}, out.getShape());
        assertArrayEquals(new double[]{
                1, 2, 1, 2, 1, 2,
                3, 4, 3, 4, 3, 4,
                1, 2, 1, 2, 1, 2,
                3, 4, 3, 4, 3, 4
        }, out.toDoubleArrayCopy(), 1e-9);
        assertTrue(containsOp(compiledGraph, Operation.OpType.TILE));
    }

    private static boolean containsOp(CompiledGraph compiledGraph, Operation.OpType opType) {
        return compiledGraph.compiledNodes().stream()
                .map(graph.CompiledNode::operation)
                .filter(op -> op != null)
                .map(Operation::opType)
                .anyMatch(type -> type == opType);
    }
}
