import graph.CompiledGraph;
import config.optimizer.OptimizerConfig;
import config.optimizer.OptimizerStage;
import operations.Operation;
import tensor.DataType;
import tensor.Tensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class AlgebraicRewritingSigmoidTest {

    @Test
    void rewritesCanonicalSigmoidNegFormInInference() {
        Tensor baselineInput = new Tensor(new double[]{0.7}, new int[]{1}, null, "x_base", DataType.FLOAT64);
        Tensor baselineOutput = baselineInput.neg().exp().add(Tensor.scalar(1.0)).inv();
        CompiledGraph.compile(baselineOutput, OptimizerConfig.noOptimization())
                .execute(config.runtime.RuntimeConfig.inferenceDefaults(), backend.runtime.ExecutionMode.FORWARD);

        Tensor optimizedInput = new Tensor(new double[]{0.7}, new int[]{1}, null, "x_opt", DataType.FLOAT64);
        Tensor optimizedOutput = optimizedInput.neg().exp().add(Tensor.scalar(1.0)).inv();
        CompiledGraph compiledGraph = CompiledGraph.compile(optimizedOutput, arOnlyInferenceConfig());
        compiledGraph.execute(config.runtime.RuntimeConfig.inferenceDefaults(), backend.runtime.ExecutionMode.FORWARD);

        assertTrue(containsSigmoid(compiledGraph),
                "Algebraic rewriting should replace canonical sigmoid form in inference");
        assertArrayEquals(baselineOutput.toDoubleArrayCopy(), optimizedOutput.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void rewritesCanonicalSigmoidMulScalarFormInInference() {
        Tensor baselineInput = new Tensor(new double[]{1.3}, new int[]{1}, null, "x_base", DataType.FLOAT64);
        Tensor baselineOutput = baselineInput.mul(-1.0).exp().add(Tensor.scalar(1.0)).inv();
        CompiledGraph.compile(baselineOutput, OptimizerConfig.noOptimization())
                .execute(config.runtime.RuntimeConfig.inferenceDefaults(), backend.runtime.ExecutionMode.FORWARD);

        Tensor optimizedInput = new Tensor(new double[]{1.3}, new int[]{1}, null, "x_opt", DataType.FLOAT64);
        Tensor optimizedOutput = optimizedInput.mul(-1.0).exp().add(Tensor.scalar(1.0)).inv();
        CompiledGraph compiledGraph = CompiledGraph.compile(optimizedOutput, arOnlyInferenceConfig());
        compiledGraph.execute(config.runtime.RuntimeConfig.inferenceDefaults(), backend.runtime.ExecutionMode.FORWARD);

        assertTrue(containsSigmoid(compiledGraph),
                "Algebraic rewriting should rewrite mulScalar(-1) sigmoid form in inference");
        assertArrayEquals(baselineOutput.toDoubleArrayCopy(), optimizedOutput.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void doesNotRewriteSigmoidWhenGradientsAreRequired() {
        Tensor baselineInput = new Tensor(new double[]{0.7}, new int[]{1}, null, "x_base", DataType.FLOAT64);
        baselineInput.setRequiresGrad(true);
        Tensor baselineOutput = baselineInput.neg().exp().add(Tensor.scalar(1.0)).inv();
        CompiledGraph.compile(baselineOutput, OptimizerConfig.noOptimization())
                .execute(config.runtime.RuntimeConfig.trainingDefaults(), backend.runtime.ExecutionMode.FORWARD_BACKWARD);

        Tensor optimizedInput = new Tensor(new double[]{0.7}, new int[]{1}, null, "x_opt", DataType.FLOAT64);
        optimizedInput.setRequiresGrad(true);
        Tensor optimizedOutput = optimizedInput.neg().exp().add(Tensor.scalar(1.0)).inv();
        CompiledGraph compiledGraph = CompiledGraph.compile(optimizedOutput, arOnlyInferenceConfig());
        compiledGraph.execute(config.runtime.RuntimeConfig.trainingDefaults(), backend.runtime.ExecutionMode.FORWARD_BACKWARD);

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
        compiledGraph.execute(config.runtime.RuntimeConfig.inferenceDefaults(), backend.runtime.ExecutionMode.FORWARD);

        assertEquals(1, compiledGraph.getCompiledGraphAsList().stream()
                .map(Tensor::getOperation)
                .filter(op -> op != null && op.opType() == Operation.OpType.GT)
                .count());
        assertArrayEquals(new byte[]{0, 1, 0}, out.getBoolData());
    }

    private static boolean containsSigmoid(CompiledGraph compiledGraph) {
        return compiledGraph.getCompiledGraphAsList().stream()
                .map(Tensor::getOperation)
                .filter(op -> op != null)
                .map(Operation::opType)
                .anyMatch(opType -> opType == Operation.OpType.SIGMOID);
    }

    private static OptimizerConfig arOnlyInferenceConfig() {
        return OptimizerConfig.inferenceDefaults().withStageOrder(java.util.List.of(OptimizerStage.AR));
    }
}
