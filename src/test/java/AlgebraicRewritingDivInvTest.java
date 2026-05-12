import config.compile.CompileConfig;
import config.compile.GraphOptimizationConfig;
import graph.CompiledGraph;
import operations.Operation;
import operations.elementwise.binary.div;
import operations.elementwise.unary.inv;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class AlgebraicRewritingDivInvTest {

    @Test
    void rewritesOneOverXToInv() {
        Tensor input = new Tensor(new double[]{2.0, 4.0}, new int[]{2}, null, "x", DataType.FLOAT64);
        Tensor one = Tensor.scalar(1.0, DataType.FLOAT64);
        Tensor divNode = new Tensor(new int[]{2}, List.of(one, input), new div(tensor.TensorBroadcastOps.planBinary(one, input)),
                "oneOverX", DataType.FLOAT64);

        CompiledGraph compiledGraph = CompiledGraph.compile(divNode, arOnlyInferenceConfig());
        compiledGraph.execute(config.runtime.RuntimeConfig.inferenceDefaults(), backend.runtime.ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{0.5, 0.25}, divNode.toDoubleArrayCopy(), 1e-9);
        assertTrue(compiledGraph.getCompiledGraphAsList().stream()
                .map(Tensor::getOperation)
                .filter(op -> op != null)
                .map(Operation::opType)
                .anyMatch(opType -> opType == Operation.OpType.INV));
        assertTrue(compiledGraph.getCompiledGraphAsList().stream()
                .map(Tensor::getOperation)
                .filter(op -> op != null)
                .map(Operation::opType)
                .noneMatch(opType -> opType == Operation.OpType.DIV));
    }

    @Test
    void rewritesInvInvToIdentity() {
        Tensor input = new Tensor(new double[]{2.0, 4.0}, new int[]{2}, null, "x", DataType.FLOAT64);
        Tensor innerInv = new Tensor(new int[]{2}, List.of(input), new inv(), "invX", DataType.FLOAT64);
        Tensor outerInv = new Tensor(new int[]{2}, List.of(innerInv), new inv(), "invInvX", DataType.FLOAT64);

        CompiledGraph compiledGraph = CompiledGraph.compile(outerInv, arOnlyInferenceConfig());
        compiledGraph.execute(config.runtime.RuntimeConfig.inferenceDefaults(), backend.runtime.ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{2.0, 4.0}, outerInv.toDoubleArrayCopy(), 1e-9);
        assertTrue(compiledGraph.getCompiledGraphAsList().stream()
                .map(Tensor::getOperation)
                .filter(op -> op != null)
                .map(Operation::opType)
                .noneMatch(opType -> opType == Operation.OpType.INV));
    }

    private static CompileConfig arOnlyInferenceConfig() {
        return CompileConfig.inference().withGraphOptimization(GraphOptimizationConfig.stages(true, false, false, false, false));
    }
}
