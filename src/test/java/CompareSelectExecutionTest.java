import backend.runtime.ExecutionMode;
import config.optimizer.OptimizerConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class CompareSelectExecutionTest {

    @Test
    void greaterThanBroadcastProducesBoolMask() {
        Tensor a = new Tensor(new double[]{1, 5, 3, 7, 2, 9}, new int[]{2, 3}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{4, 4, 4}, new int[]{3}, null, "b", DataType.FLOAT64);

        Tensor mask = a.greaterThan(b);
        CompiledGraph.compile(mask, OptimizerConfig.noOptimization()).execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertEquals(DataType.BOOL, mask.getDataType());
        assertArrayEquals(new int[]{2, 3}, mask.getShape());
        assertArrayEquals(new boolean[]{false, true, false, true, false, true}, mask.toBooleanArrayCopy());
    }

    @Test
    void equalToProducesBoolMaskForFloat32Inputs() {
        Tensor a = new Tensor(new float[]{1f, 2f, 2f, 4f}, new int[]{2, 2}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 0f, 2f, 5f}, new int[]{2, 2}, null, "b", DataType.FLOAT32);

        Tensor mask = a.equalTo(b);
        CompiledGraph.compile(mask, OptimizerConfig.noOptimization()).execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertEquals(DataType.BOOL, mask.getDataType());
        assertArrayEquals(new boolean[]{true, false, true, false}, mask.toBooleanArrayCopy());
    }

    @Test
    void whereForwardSupportsBroadcastConditionAndBranchBroadcast() {
        Tensor condition = new Tensor(new byte[]{1, 0}, new int[]{2, 1}, null, "cond", DataType.BOOL);
        Tensor ifTrue = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "x", DataType.FLOAT64);
        Tensor ifFalse = new Tensor(new double[]{10, 20, 30}, new int[]{1, 3}, null, "y", DataType.FLOAT64);

        Tensor out = Tensor.where(condition, ifTrue, ifFalse);
        CompiledGraph.compile(out, OptimizerConfig.noOptimization()).execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertEquals(DataType.FLOAT64, out.getDataType());
        assertArrayEquals(new int[]{2, 3}, out.getShape());
        assertArrayEquals(new double[]{1, 2, 3, 10, 20, 30}, out.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void whereBackwardRoutesGradientsAndReducesBroadcastedBranch() {
        Tensor condition = new Tensor(new byte[]{1, 0}, new int[]{2, 1}, null, "cond", DataType.BOOL);
        Tensor ifTrue = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "x", DataType.FLOAT64);
        Tensor ifFalse = new Tensor(new double[]{10, 20, 30}, new int[]{1, 3}, null, "y", DataType.FLOAT64);
        ifTrue.setRequiresGrad(true);
        ifFalse.setRequiresGrad(true);

        Tensor out = Tensor.where(condition, ifTrue, ifFalse);
        CompiledGraph.compile(out, OptimizerConfig.trainingDefaults()).execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(new double[]{1, 1, 1, 0, 0, 0}, ifTrue.getGradient().toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new int[]{1, 3}, ifFalse.getGradient().getShape());
        assertArrayEquals(new double[]{1, 1, 1}, ifFalse.getGradient().toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void wherePromotesNumericBranches() {
        Tensor condition = new Tensor(new byte[]{1, 0}, new int[]{2}, null, "cond", DataType.BOOL);
        Tensor ifTrue = new Tensor(new float[]{1f, 2f}, new int[]{2}, null, "x", DataType.FLOAT32);
        Tensor ifFalse = new Tensor(new double[]{10, 20}, new int[]{2}, null, "y", DataType.FLOAT64);

        Tensor out = Tensor.where(condition, ifTrue, ifFalse);
        CompiledGraph.compile(out, OptimizerConfig.noOptimization()).execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertEquals(DataType.FLOAT64, out.getDataType());
        assertArrayEquals(new double[]{1, 20}, out.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void boolRootDoesNotSupportBackwardExecution() {
        Tensor a = new Tensor(new double[]{1, 2}, new int[]{2}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{0, 3}, new int[]{2}, null, "b", DataType.FLOAT64);
        a.setRequiresGrad(true);
        b.setRequiresGrad(true);

        Tensor mask = a.greaterThan(b);

        assertThrows(UnsupportedOperationException.class, () -> CompiledGraph.compile(mask, OptimizerConfig.trainingDefaults()));
    }
}
