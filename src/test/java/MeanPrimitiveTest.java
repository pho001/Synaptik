import backend.runtime.ExecutionMode;
import config.compile.CompileConfig;
import graph.CompiledGraph;
import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class MeanPrimitiveTest {

    @Test
    void meanAxisUsesMeanPrimitiveInCompiledGraph() {
        Tensor a = new Tensor(new double[]{
                1, 2, 3,
                4, 5, 6
        }, new int[]{2, 3}, null, "matrix", DataType.FLOAT64);

        Tensor mean = a.mean(1, true);
        CompiledGraph compiledGraph = CompiledGraph.compile(mean, CompileConfig.noGraphOptimizationBaseline());
        compiledGraph.execute(config.runtime.RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{2.0, 5.0}, mean.toDoubleArrayCopy(), 1e-9);
        assertTrue(compiledGraph.compiledNodes().stream()
                .map(graph.CompiledNode::operation)
                .filter(op -> op != null)
                .map(Operation::opType)
                .anyMatch(opType -> opType == Operation.OpType.MEAN));
    }

    @Test
    void meanAllUsesMeanPrimitiveInCompiledGraph() {
        Tensor a = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{6}, null, "a", DataType.FLOAT64);

        Tensor mean = a.mean();
        CompiledGraph compiledGraph = CompiledGraph.compile(mean, CompileConfig.noGraphOptimizationBaseline());
        compiledGraph.execute(config.runtime.RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{3.5}, mean.toDoubleArrayCopy(), 1e-9);
        assertTrue(compiledGraph.compiledNodes().stream()
                .map(graph.CompiledNode::operation)
                .filter(op -> op != null)
                .map(Operation::opType)
                .anyMatch(opType -> opType == Operation.OpType.MEAN));
    }
}
