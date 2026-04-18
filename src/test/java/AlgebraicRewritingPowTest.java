import config.optimizer.OptimizerConfig;
import config.optimizer.OptimizerStage;
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
        compiledGraph.execute(config.runtime.RuntimeConfig.inferenceDefaults(), backend.runtime.ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{0.5, 0.25}, powNeg1.toDoubleArrayCopy(), 1e-9);
        assertTrue(compiledGraph.getCompiledGraphAsList().stream()
                .map(Tensor::getOperation)
                .filter(op -> op != null)
                .map(Operation::opType)
                .anyMatch(opType -> opType == Operation.OpType.INV));
        assertTrue(compiledGraph.getCompiledGraphAsList().stream()
                .map(Tensor::getOperation)
                .filter(op -> op != null)
                .map(Operation::opType)
                .noneMatch(opType -> opType == Operation.OpType.POW));
    }

    private static OptimizerConfig arOnlyInferenceConfig() {
        return OptimizerConfig.inferenceDefaults().withStageOrder(List.of(OptimizerStage.AR));
    }
}
