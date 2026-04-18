import config.optimizer.OptimizerConfig;
import config.optimizer.OptimizerStage;
import graph.CompiledGraph;
import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class AlgebraicRewritingClampTest {

    @Test
    void flattensNestedClampMinToSingleTighterThreshold() {
        Tensor input = new Tensor(new double[]{-5.0, 2.0, 7.0}, new int[]{3}, null, "x", DataType.FLOAT64);
        Tensor nested = input.clampMin(1.0).clampMin(3.0);

        CompiledGraph compiledGraph = CompiledGraph.compile(nested, arOnlyInferenceConfig());
        compiledGraph.execute(config.runtime.RuntimeConfig.inferenceDefaults(), backend.runtime.ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{3.0, 3.0, 7.0}, nested.toDoubleArrayCopy(), 1e-9);
        long clampMinCount = compiledGraph.getCompiledGraphAsList().stream()
                .map(Tensor::getOperation)
                .filter(op -> op != null && op.opType() == Operation.OpType.CLAMP_MIN)
                .count();
        assertEquals(1L, clampMinCount);
    }

    @Test
    void flattensNestedClampMaxToSingleTighterThreshold() {
        Tensor input = new Tensor(new double[]{-5.0, 2.0, 7.0}, new int[]{3}, null, "x", DataType.FLOAT64);
        Tensor nested = input.clampMax(6.0).clampMax(4.0);

        CompiledGraph compiledGraph = CompiledGraph.compile(nested, arOnlyInferenceConfig());
        compiledGraph.execute(config.runtime.RuntimeConfig.inferenceDefaults(), backend.runtime.ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{-5.0, 2.0, 4.0}, nested.toDoubleArrayCopy(), 1e-9);
        long clampMaxCount = compiledGraph.getCompiledGraphAsList().stream()
                .map(Tensor::getOperation)
                .filter(op -> op != null && op.opType() == Operation.OpType.CLAMP_MAX)
                .count();
        assertEquals(1L, clampMaxCount);
    }

    private static OptimizerConfig arOnlyInferenceConfig() {
        return OptimizerConfig.inferenceDefaults().withStageOrder(List.of(OptimizerStage.AR));
    }
}
