import config.compile.CompileConfig;
import config.compile.GraphOptimizationConfig;
import graph.CompiledGraph;
import operations.Operation;
import operations.elementwise.unary.pow;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class AlgebraicRewritingPowTest {

    @Test
    void rewritesPowNegativeOneToInv() {
        Tensor input = new Tensor(new double[]{2.0, 4.0}, new int[]{2}, null, "x", DataType.FLOAT64);
        Tensor powNeg1 = new Tensor(new int[]{2}, List.of(input), new pow(-1.0), "powNeg1", DataType.FLOAT64);

        CompiledGraph compiledGraph = CompiledGraph.compile(powNeg1, arOnlyInferenceConfig());
        compiledGraph.prepare(config.runtime.RuntimeConfig.inferenceDefaults()).execute(backend.runtime.ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{0.5, 0.25}, powNeg1.toDoubleArrayCopy(), 1e-9);
        assertTrue(compiledGraph.program().compiledNodes().stream()
                .map(graph.model.CompiledNode::operation)
                .filter(op -> op != null)
                .map(Operation::opType)
                .anyMatch(opType -> opType == Operation.OpType.INV));
        assertTrue(compiledGraph.program().compiledNodes().stream()
                .map(graph.model.CompiledNode::operation)
                .filter(op -> op != null)
                .map(Operation::opType)
                .noneMatch(opType -> opType == Operation.OpType.POW));
    }

    @Test
    void rewritesPowNegativeTwoToMulThenInv() {
        Tensor input = new Tensor(new double[]{2.0, 4.0}, new int[]{2}, null, "x", DataType.FLOAT64);
        Tensor powNeg2 = new Tensor(new int[]{2}, List.of(input), new pow(-2.0), "powNeg2", DataType.FLOAT64);

        CompiledGraph compiledGraph = CompiledGraph.compile(powNeg2, arOnlyInferenceConfig());
        compiledGraph.prepare(config.runtime.RuntimeConfig.inferenceDefaults()).execute(backend.runtime.ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{0.25, 0.0625}, powNeg2.toDoubleArrayCopy(), 1e-9);
        assertTrue(compiledGraph.program().compiledNodes().stream()
                .map(graph.model.CompiledNode::operation)
                .filter(op -> op != null)
                .map(Operation::opType)
                .anyMatch(opType -> opType == Operation.OpType.MUL));
        assertTrue(compiledGraph.program().compiledNodes().stream()
                .map(graph.model.CompiledNode::operation)
                .filter(op -> op != null)
                .map(Operation::opType)
                .anyMatch(opType -> opType == Operation.OpType.INV));
        assertTrue(compiledGraph.program().compiledNodes().stream()
                .map(graph.model.CompiledNode::operation)
                .filter(op -> op != null)
                .map(Operation::opType)
                .noneMatch(opType -> opType == Operation.OpType.POW));
    }

    private static CompileConfig arOnlyInferenceConfig() {
        return CompileConfig.inference().withGraphOptimization(GraphOptimizationConfig.stages(true, false, false, false, false));
    }
}
