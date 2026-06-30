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
import static org.junit.jupiter.api.Assertions.assertTrue;

public class ClampLoweringTest {

    @Test
    void keepsWhereLessThanFormAsSelectInCurrentArRule() {
        Tensor baselineInput = new Tensor(new double[]{-2.0, 0.0, 0.5, 3.0}, new int[]{4}, null, "x_base", DataType.FLOAT64);
        baselineInput.setRequiresGrad(true);
        Tensor lowerBase = Tensor.scalar(0.0, DataType.FLOAT64);
        Tensor baselineOut = Tensor.where(baselineInput.lessThan(lowerBase), lowerBase, baselineInput);
        CompiledGraph.compile(baselineOut, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.trainingDefaults()).execute(ExecutionMode.FORWARD_BACKWARD);

        Tensor optimizedInput = new Tensor(new double[]{-2.0, 0.0, 0.5, 3.0}, new int[]{4}, null, "x_opt", DataType.FLOAT64);
        optimizedInput.setRequiresGrad(true);
        Tensor lowerOpt = Tensor.scalar(0.0, DataType.FLOAT64);
        Tensor optimizedOut = Tensor.where(optimizedInput.lessThan(lowerOpt), lowerOpt, optimizedInput);
        CompiledGraph compiledGraph = CompiledGraph.compile(optimizedOut, arOnlyConfig());
        compiledGraph.prepare(RuntimeConfig.trainingDefaults()).execute(ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(baselineOut.toDoubleArrayCopy(), optimizedOut.toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(baselineInput.getGradient().toDoubleArrayCopy(), optimizedInput.getGradient().toDoubleArrayCopy(), 1e-9);
        assertTrue(containsOp(compiledGraph, Operation.OpType.WHERE));
        assertTrue(!containsOp(compiledGraph, Operation.OpType.CLAMP_MIN));
    }

    @Test
    void keepsWhereGreaterThanFormAsSelectInCurrentArRule() {
        Tensor baselineInput = new Tensor(new double[]{-2.0, 0.0, 0.5, 3.0}, new int[]{4}, null, "x_base", DataType.FLOAT64);
        baselineInput.setRequiresGrad(true);
        Tensor upperBase = Tensor.scalar(1.0, DataType.FLOAT64);
        Tensor baselineOut = Tensor.where(baselineInput.greaterThan(upperBase), upperBase, baselineInput);
        CompiledGraph.compile(baselineOut, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.trainingDefaults()).execute(ExecutionMode.FORWARD_BACKWARD);

        Tensor optimizedInput = new Tensor(new double[]{-2.0, 0.0, 0.5, 3.0}, new int[]{4}, null, "x_opt", DataType.FLOAT64);
        optimizedInput.setRequiresGrad(true);
        Tensor upperOpt = Tensor.scalar(1.0, DataType.FLOAT64);
        Tensor optimizedOut = Tensor.where(optimizedInput.greaterThan(upperOpt), upperOpt, optimizedInput);
        CompiledGraph compiledGraph = CompiledGraph.compile(optimizedOut, arOnlyConfig());
        compiledGraph.prepare(RuntimeConfig.trainingDefaults()).execute(ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(baselineOut.toDoubleArrayCopy(), optimizedOut.toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(baselineInput.getGradient().toDoubleArrayCopy(), optimizedInput.getGradient().toDoubleArrayCopy(), 1e-9);
        assertTrue(containsOp(compiledGraph, Operation.OpType.WHERE));
        assertTrue(!containsOp(compiledGraph, Operation.OpType.CLAMP_MAX));
    }

    @Test
    void publicClampMinAndClampMaxUseSpecializedPrimitives() {
        Tensor x = new Tensor(new double[]{-2.0, 0.0, 0.5, 3.0}, new int[]{4}, null, "x", DataType.FLOAT64);

        Tensor yMin = x.clampMin(0.0);
        Tensor yMax = x.clampMax(1.0);

        CompiledGraph minGraph = CompiledGraph.compile(yMin, CompileConfig.noGraphOptimizationBaseline());
        CompiledGraph maxGraph = CompiledGraph.compile(yMax, CompileConfig.noGraphOptimizationBaseline());
        minGraph.prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);
        maxGraph.prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        assertTrue(containsOp(minGraph, Operation.OpType.CLAMP_MIN));
        assertTrue(containsOp(maxGraph, Operation.OpType.CLAMP_MAX));
    }

    @Test
    void explicitPiecewisePolicyCanLowerClampWherePatterns() {
        Tensor xMin = new Tensor(new double[]{-2.0, 0.0, 0.5, 3.0}, new int[]{4}, null, "x_min", DataType.FLOAT64);
        Tensor minOut = Tensor.where(xMin.lessThan(Tensor.scalar(0.0, DataType.FLOAT64)), Tensor.scalar(0.0, DataType.FLOAT64), xMin);

        Tensor xMax = new Tensor(new double[]{-2.0, 0.0, 0.5, 3.0}, new int[]{4}, null, "x_max", DataType.FLOAT64);
        Tensor maxOut = Tensor.where(xMax.greaterThan(Tensor.scalar(1.0, DataType.FLOAT64)), Tensor.scalar(1.0, DataType.FLOAT64), xMax);

        CompiledGraph minGraph = CompiledGraph.compile(minOut, arWithPiecewiseConfig(new PiecewiseLoweringConfig(false, false, true)));
        CompiledGraph maxGraph = CompiledGraph.compile(maxOut, arWithPiecewiseConfig(new PiecewiseLoweringConfig(false, false, true)));
        minGraph.prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);
        maxGraph.prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        assertTrue(containsOp(minGraph, Operation.OpType.CLAMP_MIN));
        assertTrue(containsOp(maxGraph, Operation.OpType.CLAMP_MAX));
    }

    private static boolean containsOp(CompiledGraph compiledGraph, Operation.OpType opType) {
        return compiledGraph.program().compiledNodes().stream()
                .map(graph.model.CompiledNode::operation)
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
