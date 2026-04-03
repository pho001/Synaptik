import backend.runtime.ExecutionMode;
import config.optimizer.OptimizerConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class ClampExecutionTest {

    @Test
    void clampForwardClipsBelowAndAboveBounds() {
        Tensor x = new Tensor(new double[]{-2.0, -0.5, 0.5, 3.0}, new int[]{4}, null, "x", DataType.FLOAT64);

        Tensor y = x.clamp(0.0, 1.0);
        CompiledGraph.compile(y, OptimizerConfig.noOptimization()).execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{0.0, 0.0, 0.5, 1.0}, y.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void clampBackwardRoutesGradientOnlyThroughInteriorAndBoundaryTies() {
        Tensor x = new Tensor(new double[]{-2.0, 0.0, 0.5, 1.0, 3.0}, new int[]{5}, null, "x", DataType.FLOAT64);
        x.setRequiresGrad(true);

        Tensor y = x.clamp(0.0, 1.0);
        CompiledGraph.compile(y, OptimizerConfig.trainingDefaults()).execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(new double[]{0.0, 1.0, 1.0, 1.0, 0.0}, x.getGradient().toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void clampRejectsInvalidBounds() {
        Tensor x = new Tensor(new double[]{1.0}, new int[]{1}, null, "x", DataType.FLOAT64);
        assertThrows(IllegalArgumentException.class, () -> x.clamp(2.0, 1.0));
    }

    @Test
    void clampMinForwardRaisesOnlyValuesBelowLowerBound() {
        Tensor x = new Tensor(new double[]{-2.0, -0.5, 0.5, 3.0}, new int[]{4}, null, "x", DataType.FLOAT64);

        Tensor y = x.clampMin(0.0);
        CompiledGraph.compile(y, OptimizerConfig.noOptimization()).execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{0.0, 0.0, 0.5, 3.0}, y.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void clampMaxForwardLowersOnlyValuesAboveUpperBound() {
        Tensor x = new Tensor(new double[]{-2.0, -0.5, 0.5, 3.0}, new int[]{4}, null, "x", DataType.FLOAT64);

        Tensor y = x.clampMax(1.0);
        CompiledGraph.compile(y, OptimizerConfig.noOptimization()).execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{-2.0, -0.5, 0.5, 1.0}, y.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void clampMinAndClampMaxBackwardFollowSelectedBranch() {
        Tensor xMin = new Tensor(new double[]{-2.0, 0.0, 0.5, 3.0}, new int[]{4}, null, "xMin", DataType.FLOAT64);
        xMin.setRequiresGrad(true);
        Tensor yMin = xMin.clampMin(0.0);
        CompiledGraph.compile(yMin, OptimizerConfig.trainingDefaults()).execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);
        assertArrayEquals(new double[]{0.0, 1.0, 1.0, 1.0}, xMin.getGradient().toDoubleArrayCopy(), 1e-9);

        Tensor xMax = new Tensor(new double[]{-2.0, 0.0, 0.5, 3.0}, new int[]{4}, null, "xMax", DataType.FLOAT64);
        xMax.setRequiresGrad(true);
        Tensor yMax = xMax.clampMax(1.0);
        CompiledGraph.compile(yMax, OptimizerConfig.trainingDefaults()).execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);
        assertArrayEquals(new double[]{1.0, 1.0, 1.0, 0.0}, xMax.getGradient().toDoubleArrayCopy(), 1e-9);
    }
}
