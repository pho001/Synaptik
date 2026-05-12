import backend.runtime.ExecutionMode;
import config.optimizer.PiecewiseLoweringConfig;
import config.compile.CompileConfig;
import config.compile.GraphOptimizationConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class ReluLoweringTest {

    @Test
    void keepsStrictPositiveWherePatternAsSelectInCurrentArRule() {
        Tensor baselineInput = new Tensor(new double[]{-2.0, 0.0, 3.0}, new int[]{3}, null, "x_base", DataType.FLOAT64);
        baselineInput.setRequiresGrad(true);
        Tensor baselineOut = Tensor.where(
                baselineInput.greaterThan(Tensor.scalar(0.0, DataType.FLOAT64)),
                baselineInput,
                Tensor.zerosLike(baselineInput)
        );
        CompiledGraph.compile(baselineOut, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);

        Tensor optimizedInput = new Tensor(new double[]{-2.0, 0.0, 3.0}, new int[]{3}, null, "x_opt", DataType.FLOAT64);
        optimizedInput.setRequiresGrad(true);
        Tensor optimizedOut = Tensor.where(
                optimizedInput.greaterThan(Tensor.scalar(0.0, DataType.FLOAT64)),
                optimizedInput,
                Tensor.zerosLike(optimizedInput)
        );
        CompiledGraph compiledGraph = CompiledGraph.compile(optimizedOut, arOnlyConfig());
        compiledGraph.execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(baselineOut.toDoubleArrayCopy(), optimizedOut.toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(baselineInput.getGradient().toDoubleArrayCopy(), optimizedInput.getGradient().toDoubleArrayCopy(), 1e-9);
        assertFalse(containsOp(compiledGraph, Operation.OpType.RELU));
        assertTrue(containsOp(compiledGraph, Operation.OpType.WHERE));
    }

    @Test
    void doesNotLowerGreaterOrEqualVariantToRelu() {
        Tensor input = new Tensor(new double[]{-2.0, 0.0, 3.0}, new int[]{3}, null, "x", DataType.FLOAT64);
        Tensor out = Tensor.where(
                input.greaterOrEqual(Tensor.scalar(0.0, DataType.FLOAT64)),
                input,
                Tensor.zerosLike(input)
        );

        CompiledGraph compiledGraph = CompiledGraph.compile(out, arOnlyConfig());
        compiledGraph.execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertFalse(containsOp(compiledGraph, Operation.OpType.RELU));
        assertTrue(containsOp(compiledGraph, Operation.OpType.WHERE));
    }

    @Test
    void minimumAndMaximumAreNotLoweredToSpecializedMinMax() {
        Tensor a = new Tensor(new double[]{1.0, 5.0, 3.0}, new int[]{3}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{2.0, 4.0, 3.0}, new int[]{3}, null, "b", DataType.FLOAT64);

        Tensor min = a.minimum(b);
        Tensor max = a.maximum(b);

        CompiledGraph minGraph = CompiledGraph.compile(min, arOnlyConfig());
        CompiledGraph maxGraph = CompiledGraph.compile(max, arOnlyConfig());
        minGraph.execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
        maxGraph.execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertFalse(containsOp(minGraph, Operation.OpType.MIN));
        assertFalse(containsOp(maxGraph, Operation.OpType.MAX));
        assertTrue(containsOp(minGraph, Operation.OpType.WHERE));
        assertTrue(containsOp(maxGraph, Operation.OpType.WHERE));
    }

    @Test
    void publicReluMatchesExpectedForwardAndBackward() {
        Tensor input = new Tensor(new double[]{-2.0, 0.0, 3.0}, new int[]{3}, null, "x", DataType.FLOAT64);
        input.setRequiresGrad(true);
        Tensor out = input.relu();

        CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                .execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(new double[]{0.0, 0.0, 3.0}, out.toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{0.0, 0.0, 1.0}, input.getGradient().toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void explicitPiecewisePolicyCanLowerReluLikeWherePattern() {
        Tensor input = new Tensor(new double[]{-2.0, 0.0, 3.0}, new int[]{3}, null, "x", DataType.FLOAT64);
        Tensor out = Tensor.where(
                input.greaterThan(Tensor.scalar(0.0, DataType.FLOAT64)),
                input,
                Tensor.zerosLike(input)
        );

        CompiledGraph compiledGraph = CompiledGraph.compile(
                out,
                arWithPiecewiseConfig(new PiecewiseLoweringConfig(false, true, false))
        );
        compiledGraph.execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertTrue(containsOp(compiledGraph, Operation.OpType.RELU));
    }

    private static boolean containsOp(CompiledGraph compiledGraph, Operation.OpType opType) {
        return compiledGraph.getCompiledGraphAsList().stream()
                .map(Tensor::getOperation)
                .filter(op -> op != null)
                .map(Operation::opType)
                .anyMatch(type -> type == opType);
    }

    private static CompileConfig arOnlyConfig() {
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
