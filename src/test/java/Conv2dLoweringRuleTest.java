import config.compile.CompileConfig;
import config.compile.GraphOptimizationConfig;
import config.optimizer.Conv2dDagLoweringProfile;
import config.optimizer.Conv2dLoweringConfig;
import config.optimizer.Conv2dLoweringMode;
import config.optimizer.RewriteConfig;
import graph.CompiledGraph;
import graph.optimizer.GraphOptimizer;
import graph.optimizer.rewrite.lowering.Conv2dDagLoweringRule;
import org.junit.jupiter.api.Test;
import operations.Operation;
import tensor.CompileMode;
import tensor.options.Conv2dOptions;
import tensor.DataType;
import tensor.Tensor;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class Conv2dLoweringRuleTest {
    @Test
    void lowersForwardConv2dToWindowMatmulDag() {
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
        GraphOptimizer optimizer = new GraphOptimizer().addRule(new Conv2dDagLoweringRule(Conv2dLoweringConfig.always()));
        List<Tensor> optimized = optimizer.optimize(root.topologicalSort());

        assertEquals(0, countOp(optimized, Operation.OpType.CONV2D));
        assertTrue(countOp(optimized, Operation.OpType.UNFOLD2D) >= 1);
        assertTrue(countOp(optimized, Operation.OpType.MATMUL) >= 1);
        assertTrue(countOp(optimized, Operation.OpType.RESHAPE) >= 1);
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
        ).prepare(config.runtime.RuntimeConfig.inferenceDefaults()).execute(backend.runtime.ExecutionMode.FORWARD);

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
        GraphOptimizer optimizer = new GraphOptimizer().addRule(new Conv2dDagLoweringRule(
                new Conv2dLoweringConfig(Conv2dLoweringMode.OFF)
        ));
        List<Tensor> optimized = optimizer.optimize(root.topologicalSort());

        assertEquals(1, countOp(optimized, Operation.OpType.CONV2D));
        assertEquals(0, countOp(optimized, Operation.OpType.UNFOLD2D));
        assertEquals(0, countOp(optimized, Operation.OpType.MATMUL));
    }

    @Test
    void heuristicDoesNotLowerDepthwiseConv2d() {
        Tensor input = new Tensor(new double[1 * 64 * 16 * 16], new int[]{1, 64, 16, 16}, null, "input", DataType.FLOAT64);
        Tensor weight = new Tensor(new double[64 * 1 * 3 * 3], new int[]{64, 1, 3, 3}, null, "weight", DataType.FLOAT64);
        Tensor root = input.conv2d(weight, new Conv2dOptions(1, 1, 1, 1, 1, 1, 64));

        GraphOptimizer optimizer = new GraphOptimizer().addRule(new Conv2dDagLoweringRule(
                new Conv2dLoweringConfig(Conv2dLoweringMode.HEURISTIC)
        ));
        List<Tensor> optimized = optimizer.optimize(root.topologicalSort());

        assertEquals(1, countOp(optimized, Operation.OpType.CONV2D));
        assertEquals(0, countOp(optimized, Operation.OpType.UNFOLD2D));
        assertEquals(0, countOp(optimized, Operation.OpType.MATMUL));
    }

    @Test
    void heuristicLowersLargePointwiseProjectionConv2d() {
        Tensor input = new Tensor(new double[2 * 256 * 14 * 14], new int[]{2, 256, 14, 14}, null, "input", DataType.FLOAT64);
        Tensor weight = new Tensor(new double[256 * 256], new int[]{256, 256, 1, 1}, null, "weight", DataType.FLOAT64);
        Tensor root = input.conv2d(weight, Conv2dOptions.defaults());

        GraphOptimizer optimizer = new GraphOptimizer().addRule(new Conv2dDagLoweringRule(
                new Conv2dLoweringConfig(Conv2dLoweringMode.HEURISTIC)
        ));
        List<Tensor> optimized = optimizer.optimize(root.topologicalSort());

        assertEquals(0, countOp(optimized, Operation.OpType.CONV2D));
        assertTrue(countOp(optimized, Operation.OpType.UNFOLD2D) >= 1);
        assertTrue(countOp(optimized, Operation.OpType.MATMUL) >= 1);
    }

    @Test
    void heuristicUsesConfiguredDagLoweringProfileThresholds() {
        Tensor input = new Tensor(new double[1 * 64 * 8 * 8], new int[]{1, 64, 8, 8}, null, "input", DataType.FLOAT64);
        Tensor weight = new Tensor(new double[64 * 64], new int[]{64, 64, 1, 1}, null, "weight", DataType.FLOAT64);
        Tensor root = input.conv2d(weight, Conv2dOptions.defaults());

        Conv2dDagLoweringProfile permissive = new Conv2dDagLoweringProfile(
                1,
                1,
                2.0d,
                1L,
                1,
                1,
                1L
        );
        GraphOptimizer optimizer = new GraphOptimizer().addRule(new Conv2dDagLoweringRule(
                new Conv2dLoweringConfig(Conv2dLoweringMode.HEURISTIC, permissive)
        ));
        List<Tensor> optimized = optimizer.optimize(root.topologicalSort());

        assertEquals(0, countOp(optimized, Operation.OpType.CONV2D));
        assertTrue(countOp(optimized, Operation.OpType.UNFOLD2D) >= 1);
        assertTrue(countOp(optimized, Operation.OpType.MATMUL) >= 1);
    }

    @Test
    void backwardConv2dUsesCanonicalWindowMatmulDag() {
        Tensor input = new Tensor(new double[2 * 64 * 16 * 16], new int[]{2, 64, 16, 16}, null, "input", DataType.FLOAT64);
        Tensor weight = new Tensor(new double[64 * 64 * 3 * 3], new int[]{64, 64, 3, 3}, null, "weight", DataType.FLOAT64);
        input.setRequiresGrad(true);
        weight.setRequiresGrad(true);

        Tensor loss = input.conv2d(weight, Conv2dOptions.defaults().withPadding(1, 1)).sum();
        CompiledGraph compiled = CompiledGraph.compile(
                loss,
                new GraphOptimizer().addRule(new Conv2dDagLoweringRule(Conv2dLoweringConfig.always())),
                CompileMode.AUTO
        );

        assertFalse(containsOpNamePrefix(compiled, "CONV2D_" + "BACKWARD"));
        assertTrue(containsOp(compiled, operations.Operation.OpType.UNFOLD2D));
        assertTrue(containsOp(compiled, operations.Operation.OpType.FOLD2D));
        assertTrue(containsOp(compiled, operations.Operation.OpType.MATMUL));
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

        assertFalse(containsOpNamePrefix(compiled, "CONV2D_" + "BACKWARD"));
        assertTrue(containsOp(compiled, operations.Operation.OpType.UNFOLD2D));
        assertTrue(containsOp(compiled, operations.Operation.OpType.FOLD2D));
        assertTrue(containsOp(compiled, operations.Operation.OpType.MATMUL));
    }

    private static boolean containsOp(CompiledGraph compiled, operations.Operation.OpType opType) {
        return compiled.program().compiledNodes().stream()
                .anyMatch(t -> t.operation() != null && t.operation().opType() == opType);
    }

    private static long countOp(List<Tensor> tensors, Operation.OpType opType) {
        return tensors.stream()
                .filter(t -> t.getOperation() != null && t.getOperation().opType() == opType)
                .count();
    }

    private static boolean containsOpNamePrefix(CompiledGraph compiled, String prefix) {
        return compiled.program().compiledNodes().stream()
                .anyMatch(t -> t.operation() != null && t.operation().opType().name().startsWith(prefix));
    }
}
