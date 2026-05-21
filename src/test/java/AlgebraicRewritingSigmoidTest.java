import graph.CompiledGraph;
import config.optimizer.PiecewiseLoweringConfig;
import config.compile.CompileConfig;
import config.compile.GraphOptimizationConfig;
import operations.Operation;
import tensor.DataType;
import tensor.Tensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class AlgebraicRewritingSigmoidTest {

    @Test
    void doesNotRewriteCanonicalSigmoidNegFormInCurrentArRule() {
        Tensor baselineInput = new Tensor(new double[]{0.7}, new int[]{1}, null, "x_base", DataType.FLOAT64);
        Tensor baselineOutput = baselineInput.neg().exp().add(Tensor.scalar(1.0)).inv();
        CompiledGraph.compile(baselineOutput, CompileConfig.noGraphOptimizationBaseline())
                .prepare(config.runtime.RuntimeConfig.inferenceDefaults()).execute(backend.runtime.ExecutionMode.FORWARD);

        Tensor optimizedInput = new Tensor(new double[]{0.7}, new int[]{1}, null, "x_opt", DataType.FLOAT64);
        Tensor optimizedOutput = optimizedInput.neg().exp().add(Tensor.scalar(1.0)).inv();
        CompiledGraph compiledGraph = CompiledGraph.compile(optimizedOutput, arOnlyInferenceConfig());
        compiledGraph.prepare(config.runtime.RuntimeConfig.inferenceDefaults()).execute(backend.runtime.ExecutionMode.FORWARD);

        assertFalse(containsSigmoid(compiledGraph),
                "Current algebraic rewrite keeps the canonical sigmoid decomposition unchanged.");
        assertArrayEquals(baselineOutput.toDoubleArrayCopy(), optimizedOutput.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void doesNotRewriteCanonicalSigmoidMulScalarFormInCurrentArRule() {
        Tensor baselineInput = new Tensor(new double[]{1.3}, new int[]{1}, null, "x_base", DataType.FLOAT64);
        Tensor baselineOutput = baselineInput.mul(-1.0).exp().add(Tensor.scalar(1.0)).inv();
        CompiledGraph.compile(baselineOutput, CompileConfig.noGraphOptimizationBaseline())
                .prepare(config.runtime.RuntimeConfig.inferenceDefaults()).execute(backend.runtime.ExecutionMode.FORWARD);

        Tensor optimizedInput = new Tensor(new double[]{1.3}, new int[]{1}, null, "x_opt", DataType.FLOAT64);
        Tensor optimizedOutput = optimizedInput.mul(-1.0).exp().add(Tensor.scalar(1.0)).inv();
        CompiledGraph compiledGraph = CompiledGraph.compile(optimizedOutput, arOnlyInferenceConfig());
        compiledGraph.prepare(config.runtime.RuntimeConfig.inferenceDefaults()).execute(backend.runtime.ExecutionMode.FORWARD);

        assertFalse(containsSigmoid(compiledGraph),
                "Current algebraic rewrite keeps the mulScalar(-1) sigmoid decomposition unchanged.");
        assertArrayEquals(baselineOutput.toDoubleArrayCopy(), optimizedOutput.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void explicitPiecewisePolicyCanLowerCanonicalSigmoid() {
        Tensor input = new Tensor(new double[]{0.7}, new int[]{1}, null, "x_opt", DataType.FLOAT64);
        Tensor output = input.neg().exp().add(Tensor.scalar(1.0)).inv();
        CompiledGraph compiledGraph = CompiledGraph.compile(
                output,
                arWithPiecewiseConfig(new PiecewiseLoweringConfig(true, false, false))
        );
        compiledGraph.prepare(config.runtime.RuntimeConfig.inferenceDefaults()).execute(backend.runtime.ExecutionMode.FORWARD);

        assertTrue(containsSigmoid(compiledGraph));
    }

    @Test
    void doesNotRewriteSigmoidWhenGradientsAreRequired() {
        Tensor baselineInput = new Tensor(new double[]{0.7}, new int[]{1}, null, "x_base", DataType.FLOAT64);
        baselineInput.setRequiresGrad(true);
        Tensor baselineOutput = baselineInput.neg().exp().add(Tensor.scalar(1.0)).inv();
        CompiledGraph.compile(baselineOutput, CompileConfig.noGraphOptimizationBaseline())
                .prepare(config.runtime.RuntimeConfig.trainingDefaults()).execute(backend.runtime.ExecutionMode.FORWARD_BACKWARD);

        Tensor optimizedInput = new Tensor(new double[]{0.7}, new int[]{1}, null, "x_opt", DataType.FLOAT64);
        optimizedInput.setRequiresGrad(true);
        Tensor optimizedOutput = optimizedInput.neg().exp().add(Tensor.scalar(1.0)).inv();
        CompiledGraph compiledGraph = CompiledGraph.compile(optimizedOutput, arOnlyInferenceConfig());
        compiledGraph.prepare(config.runtime.RuntimeConfig.trainingDefaults()).execute(backend.runtime.ExecutionMode.FORWARD_BACKWARD);

        assertTrue(!containsSigmoid(compiledGraph),
                "Sigmoid rewrite should be skipped when gradients are required");
        assertArrayEquals(baselineOutput.toDoubleArrayCopy(), optimizedOutput.toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(baselineInput.getGradient().toDoubleArrayCopy(), optimizedInput.getGradient().toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void doesNotRewriteCompareSurface() {
        Tensor x = new Tensor(new double[]{1.0, 3.0, 2.0}, new int[]{3}, null, "x", DataType.FLOAT64);
        Tensor y = new Tensor(new double[]{2.0, 2.0, 2.0}, new int[]{3}, null, "y", DataType.FLOAT64);
        Tensor out = x.greaterThan(y);

        CompiledGraph compiledGraph = CompiledGraph.compile(out, arOnlyInferenceConfig());
        compiledGraph.prepare(config.runtime.RuntimeConfig.inferenceDefaults()).execute(backend.runtime.ExecutionMode.FORWARD);

        assertEquals(1, compiledGraph.program().compiledNodes().stream()
                .map(graph.CompiledNode::operation)
                .filter(op -> op != null && op.opType() == Operation.OpType.GT)
                .count());
        assertArrayEquals(new byte[]{0, 1, 0}, out.toBoolByteArrayCopy());
    }

    private static boolean containsSigmoid(CompiledGraph compiledGraph) {
        return compiledGraph.program().compiledNodes().stream()
                .map(graph.CompiledNode::operation)
                .filter(op -> op != null)
                .map(Operation::opType)
                .anyMatch(opType -> opType == Operation.OpType.SIGMOID);
    }

    private static CompileConfig arOnlyInferenceConfig() {
        return CompileConfig.inference().withGraphOptimization(GraphOptimizationConfig.stages(true, false, false, false, false));
    }

    private static CompileConfig arWithPiecewiseConfig(PiecewiseLoweringConfig piecewiseLowering) {
        return CompileConfig.inference()
                .withGraphOptimization(GraphOptimizationConfig
                        .stages(true, false, false, false, false)
                        .withRewrite(CompileConfig.inference()
                                .graphOptimization()
                                .rewrite()
                                .withPiecewiseLowering(piecewiseLowering)));
    }
}
