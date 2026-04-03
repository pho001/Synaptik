import backend.runtime.ExecutionMode;
import config.optimizer.OptimizerConfig;
import config.optimizer.OptimizerStage;
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
    void lowersWhereLessThanFormToClampMin() {
        Tensor baselineInput = new Tensor(new double[]{-2.0, 0.0, 0.5, 3.0}, new int[]{4}, null, "x_base", DataType.FLOAT64);
        baselineInput.setRequiresGrad(true);
        Tensor lowerBase = Tensor.scalar(0.0, DataType.FLOAT64);
        Tensor baselineOut = Tensor.where(baselineInput.lessThan(lowerBase), lowerBase, baselineInput);
        CompiledGraph.compile(baselineOut, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);

        Tensor optimizedInput = new Tensor(new double[]{-2.0, 0.0, 0.5, 3.0}, new int[]{4}, null, "x_opt", DataType.FLOAT64);
        optimizedInput.setRequiresGrad(true);
        Tensor lowerOpt = Tensor.scalar(0.0, DataType.FLOAT64);
        Tensor optimizedOut = Tensor.where(optimizedInput.lessThan(lowerOpt), lowerOpt, optimizedInput);
        CompiledGraph compiledGraph = CompiledGraph.compile(optimizedOut, arOnlyConfig());
        compiledGraph.execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(baselineOut.toDoubleArrayCopy(), optimizedOut.toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(baselineInput.getGradient().toDoubleArrayCopy(), optimizedInput.getGradient().toDoubleArrayCopy(), 1e-9);
        assertTrue(containsOp(compiledGraph, Operation.OpType.CLAMP_MIN));
    }

    @Test
    void lowersWhereGreaterThanFormToClampMax() {
        Tensor baselineInput = new Tensor(new double[]{-2.0, 0.0, 0.5, 3.0}, new int[]{4}, null, "x_base", DataType.FLOAT64);
        baselineInput.setRequiresGrad(true);
        Tensor upperBase = Tensor.scalar(1.0, DataType.FLOAT64);
        Tensor baselineOut = Tensor.where(baselineInput.greaterThan(upperBase), upperBase, baselineInput);
        CompiledGraph.compile(baselineOut, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);

        Tensor optimizedInput = new Tensor(new double[]{-2.0, 0.0, 0.5, 3.0}, new int[]{4}, null, "x_opt", DataType.FLOAT64);
        optimizedInput.setRequiresGrad(true);
        Tensor upperOpt = Tensor.scalar(1.0, DataType.FLOAT64);
        Tensor optimizedOut = Tensor.where(optimizedInput.greaterThan(upperOpt), upperOpt, optimizedInput);
        CompiledGraph compiledGraph = CompiledGraph.compile(optimizedOut, arOnlyConfig());
        compiledGraph.execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(baselineOut.toDoubleArrayCopy(), optimizedOut.toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(baselineInput.getGradient().toDoubleArrayCopy(), optimizedInput.getGradient().toDoubleArrayCopy(), 1e-9);
        assertTrue(containsOp(compiledGraph, Operation.OpType.CLAMP_MAX));
    }

    @Test
    void publicClampMinAndClampMaxUseSpecializedPrimitives() {
        Tensor x = new Tensor(new double[]{-2.0, 0.0, 0.5, 3.0}, new int[]{4}, null, "x", DataType.FLOAT64);

        Tensor yMin = x.clampMin(0.0);
        Tensor yMax = x.clampMax(1.0);

        CompiledGraph minGraph = CompiledGraph.compile(yMin, OptimizerConfig.noOptimization());
        CompiledGraph maxGraph = CompiledGraph.compile(yMax, OptimizerConfig.noOptimization());
        minGraph.execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
        maxGraph.execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertTrue(containsOp(minGraph, Operation.OpType.CLAMP_MIN));
        assertTrue(containsOp(maxGraph, Operation.OpType.CLAMP_MAX));
    }

    private static boolean containsOp(CompiledGraph compiledGraph, Operation.OpType opType) {
        return compiledGraph.getCompiledGraphAsList().stream()
                .map(Tensor::getOperation)
                .filter(op -> op != null)
                .map(Operation::opType)
                .anyMatch(type -> type == opType);
    }

    private static OptimizerConfig arOnlyConfig() {
        return OptimizerConfig.inferenceDefaults().withStageOrder(List.of(OptimizerStage.AR));
    }
}
