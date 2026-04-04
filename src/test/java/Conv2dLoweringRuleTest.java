import config.optimizer.OptimizerConfig;
import config.optimizer.OptimizerStage;
import graph.CompiledGraph;
import graph.optimizer.GraphOptimizer;
import graph.optimizer.rewrite.Conv2dLoweringRewrite;
import org.junit.jupiter.api.Test;
import tensor.Conv2dOptions;
import tensor.DataType;
import tensor.Tensor;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

public class Conv2dLoweringRuleTest {
    @Test
    void lowersForwardConv2dToConv2dGemmPrimitive() {
        Tensor input = new Tensor(new double[]{
                1, 2, 3,
                4, 5, 6,
                7, 8, 9
        }, new int[]{1, 1, 3, 3}, null, "input", DataType.FLOAT64);
        Tensor weight = new Tensor(new double[]{
                1, 0,
                0, -1
        }, new int[]{1, 1, 2, 2}, null, "weight", DataType.FLOAT64);

        Tensor root = input.conv2d(weight, Conv2dOptions.defaults());
        GraphOptimizer optimizer = new GraphOptimizer().addRule(new Conv2dLoweringRewrite());
        List<Tensor> optimized = optimizer.optimize(root.topologicalSort());

        Tensor lowered = optimized.stream()
                .filter(t -> t.getOperation() != null && t.getOperation().opType() == operations.Operation.OpType.CONV2D_GEMM)
                .findFirst()
                .orElse(null);
        assertNotNull(lowered);
        assertEquals(2, lowered.getPrevTensors().size());
    }

    @Test
    void arStageConv2dLoweringPreservesForwardResult() {
        Tensor input = new Tensor(new double[]{
                1, 2, 3,
                4, 5, 6,
                7, 8, 9
        }, new int[]{1, 1, 3, 3}, null, "input", DataType.FLOAT64);
        Tensor weight = new Tensor(new double[]{
                1, 0,
                0, -1
        }, new int[]{1, 1, 2, 2}, null, "weight", DataType.FLOAT64);

        Tensor out = input.conv2d(weight, Conv2dOptions.defaults());
        CompiledGraph.compile(
                out,
                new OptimizerConfig(
                        List.of(OptimizerStage.AR),
                        config.optimizer.CseConfig.strictDefaults(),
                        config.optimizer.FuseConfig.trainingDefaults()
                )
        ).execute(config.runtime.RuntimeConfig.inferenceDefaults(), backend.runtime.ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{
                -4, -4,
                -4, -4
        }, out.toDoubleArrayCopy(), 1e-9);
    }
}
