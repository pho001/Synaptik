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

public class AbsExecutionTest {

    @Test
    void absForwardReturnsAbsoluteValues() {
        Tensor x = new Tensor(new double[]{-3.0, -0.5, 0.0, 2.0}, new int[]{4}, null, "x", DataType.FLOAT64);
        Tensor y = x.abs();

        CompiledGraph compiledGraph = CompiledGraph.compile(y, CompileConfig.noGraphOptimizationBaseline());
        compiledGraph.execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{3.0, 0.5, 0.0, 2.0}, y.toDoubleArrayCopy(), 1e-9);
        assertTrue(containsOp(compiledGraph, Operation.OpType.ABS));
    }

    @Test
    void absBackwardUsesSignWithZeroGradientAtOrigin() {
        Tensor x = new Tensor(new double[]{-3.0, -0.5, 0.0, 2.0}, new int[]{4}, null, "x", DataType.FLOAT64);
        x.setRequiresGrad(true);
        Tensor y = x.abs();

        CompiledGraph.compile(y, CompileConfig.training())
                .execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(new double[]{-1.0, -1.0, 0.0, 1.0}, x.getGradient().toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void absSupportsFloat32() {
        Tensor x = new Tensor(new float[]{-3.0f, -0.5f, 0.0f, 2.0f}, new int[]{4}, null, "x", DataType.FLOAT32);
        Tensor y = x.abs();

        CompiledGraph.compile(y, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{3.0, 0.5, 0.0, 2.0}, y.toDoubleArrayCopy(), 1e-6);
    }

    private static boolean containsOp(CompiledGraph compiledGraph, Operation.OpType opType) {
        return compiledGraph.compiledNodes().stream()
                .map(graph.CompiledNode::operation)
                .filter(op -> op != null)
                .map(Operation::opType)
                .anyMatch(type -> type == opType);
    }
}
