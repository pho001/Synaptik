import config.compile.CompileConfig;
import config.compile.GraphOptimizationConfig;
import config.optimizer.Conv2dLoweringConfig;
import config.optimizer.Conv2dLoweringMode;
import config.optimizer.RewriteConfig;
import graph.CompiledGraph;
import graph.optimizer.GraphOptimizer;
import graph.optimizer.rewrite.lowering.Conv2dGemmLoweringRule;
import org.junit.jupiter.api.Test;
import tensor.CompileMode;
import tensor.options.Conv2dOptions;
import tensor.DataType;
import tensor.Tensor;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

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
        GraphOptimizer optimizer = new GraphOptimizer().addRule(new Conv2dGemmLoweringRule(Conv2dLoweringConfig.always()));
        List<Tensor> optimized = optimizer.optimize(root.topologicalSort());

        Tensor lowered = optimized.stream()
                .filter(t -> t.getOperation() != null && t.getOperation().opType() == operations.Operation.OpType.CONV2D_GEMM)
                .findFirst()
                .orElse(null);
        assertNotNull(lowered);
        assertEquals(2, lowered.getPrevTensors().size());
    }

    @Test
    void arStageNoLongerLowersConv2dButPreservesForwardResult() {
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
                CompileConfig.inference()
                        .withGraphOptimization(GraphOptimizationConfig
                                .stages(true, false, false, false, false)
                                .withRewrite(new RewriteConfig(Conv2dLoweringConfig.always())))
        ).execute(config.runtime.RuntimeConfig.inferenceDefaults(), backend.runtime.ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{
                -4, -4,
                -4, -4
        }, out.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void rewriteConfigOffKeepsConv2dPrimitive() {
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
        GraphOptimizer optimizer = new GraphOptimizer().addRule(new Conv2dGemmLoweringRule(
                new Conv2dLoweringConfig(Conv2dLoweringMode.OFF)
        ));
        List<Tensor> optimized = optimizer.optimize(root.topologicalSort());

        long gemmCount = optimized.stream()
                .filter(t -> t.getOperation() != null && t.getOperation().opType() == operations.Operation.OpType.CONV2D_GEMM)
                .count();
        assertEquals(0, gemmCount);
    }

    @Test
    void heuristicDoesNotLowerDepthwiseConv2d() {
        Tensor input = new Tensor(new double[1 * 64 * 16 * 16], new int[]{1, 64, 16, 16}, null, "input", DataType.FLOAT64);
        Tensor weight = new Tensor(new double[64 * 1 * 3 * 3], new int[]{64, 1, 3, 3}, null, "weight", DataType.FLOAT64);
        Tensor root = input.conv2d(weight, new Conv2dOptions(1, 1, 1, 1, 1, 1, 64));

        GraphOptimizer optimizer = new GraphOptimizer().addRule(new Conv2dGemmLoweringRule(
                new Conv2dLoweringConfig(Conv2dLoweringMode.HEURISTIC)
        ));
        List<Tensor> optimized = optimizer.optimize(root.topologicalSort());

        long gemmCount = optimized.stream()
                .filter(t -> t.getOperation() != null && t.getOperation().opType() == operations.Operation.OpType.CONV2D_GEMM)
                .count();
        assertEquals(0, gemmCount);
    }

    @Test
    void heuristicLowersLargePointwiseProjectionConv2d() {
        Tensor input = new Tensor(new double[2 * 256 * 14 * 14], new int[]{2, 256, 14, 14}, null, "input", DataType.FLOAT64);
        Tensor weight = new Tensor(new double[256 * 256], new int[]{256, 256, 1, 1}, null, "weight", DataType.FLOAT64);
        Tensor root = input.conv2d(weight, Conv2dOptions.defaults());

        GraphOptimizer optimizer = new GraphOptimizer().addRule(new Conv2dGemmLoweringRule(
                new Conv2dLoweringConfig(Conv2dLoweringMode.HEURISTIC)
        ));
        List<Tensor> optimized = optimizer.optimize(root.topologicalSort());

        long gemmCount = optimized.stream()
                .filter(t -> t.getOperation() != null && t.getOperation().opType() == operations.Operation.OpType.CONV2D_GEMM)
                .count();
        assertEquals(1, gemmCount);
    }

    @Test
    void lowersBackwardConv2dPrimitivesToExplicitGemmVariants() {
        Tensor input = new Tensor(new double[2 * 64 * 16 * 16], new int[]{2, 64, 16, 16}, null, "input", DataType.FLOAT64);
        Tensor weight = new Tensor(new double[64 * 64 * 3 * 3], new int[]{64, 64, 3, 3}, null, "weight", DataType.FLOAT64);
        input.setRequiresGrad(true);
        weight.setRequiresGrad(true);

        Tensor loss = input.conv2d(weight, Conv2dOptions.defaults().withPadding(1, 1)).sum();
        CompiledGraph compiled = CompiledGraph.compile(
                loss,
                new GraphOptimizer().addRule(new Conv2dGemmLoweringRule(Conv2dLoweringConfig.always())),
                CompileMode.AUTO
        );

        boolean hasBackwardInputGemm = compiled.compiledNodes().stream()
                .anyMatch(t -> t.operation() != null && t.operation().opType() == operations.Operation.OpType.CONV2D_BACKWARD_INPUT_GEMM);
        boolean hasBackwardWeightGemm = compiled.compiledNodes().stream()
                .anyMatch(t -> t.operation() != null && t.operation().opType() == operations.Operation.OpType.CONV2D_BACKWARD_WEIGHT_GEMM);

        assertTrue(hasBackwardInputGemm);
        assertTrue(hasBackwardWeightGemm);
    }

    @Test
    void arStageNoLongerLowersBackwardConv2dToGemmVariants() {
        Tensor input = new Tensor(new double[2 * 8 * 4 * 4], new int[]{2, 8, 4, 4}, null, "input", DataType.FLOAT64);
        Tensor weight = new Tensor(new double[8 * 8 * 3 * 3], new int[]{8, 8, 3, 3}, null, "weight", DataType.FLOAT64);
        input.setRequiresGrad(true);
        weight.setRequiresGrad(true);

        Tensor loss = input.conv2d(weight, Conv2dOptions.defaults().withPadding(1, 1)).sum();
        CompiledGraph compiled = CompiledGraph.compile(
                loss,
                CompileConfig.training().withGraphOptimization(GraphOptimizationConfig.stages(true, false, false, false, false))
        );

        boolean hasBackwardInputGemm = compiled.compiledNodes().stream()
                .anyMatch(t -> t.operation() != null && t.operation().opType() == operations.Operation.OpType.CONV2D_BACKWARD_INPUT_GEMM);
        boolean hasBackwardWeightGemm = compiled.compiledNodes().stream()
                .anyMatch(t -> t.operation() != null && t.operation().opType() == operations.Operation.OpType.CONV2D_BACKWARD_WEIGHT_GEMM);

        assertFalse(hasBackwardInputGemm);
        assertFalse(hasBackwardWeightGemm);
    }
}
