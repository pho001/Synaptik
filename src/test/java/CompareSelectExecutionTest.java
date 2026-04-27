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
    void greaterOrEqualLessOrEqualAndNotEqualToWork() {
        Tensor a = new Tensor(new double[]{1, 2, 2, 4}, new int[]{2, 2}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{1, 3, 2, 0}, new int[]{2, 2}, null, "b", DataType.FLOAT64);

        Tensor ge = a.greaterOrEqual(b);
        Tensor le = a.lessOrEqual(b);
        Tensor ne = a.notEqualTo(b);

        CompiledGraph.compile(ge, OptimizerConfig.noOptimization()).execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
        CompiledGraph.compile(le, OptimizerConfig.noOptimization()).execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
        CompiledGraph.compile(ne, OptimizerConfig.noOptimization()).execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new boolean[]{true, false, true, true}, ge.toBooleanArrayCopy());
        assertArrayEquals(new boolean[]{true, true, true, false}, le.toBooleanArrayCopy());
        assertArrayEquals(new boolean[]{false, true, false, true}, ne.toBooleanArrayCopy());
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
    void whereSupportsPromotionAcrossAllNumericBranchPairs() {
        Tensor condition = new Tensor(new byte[]{1, 0}, new int[]{2}, null, "cond", DataType.BOOL);

        Tensor f16 = new Tensor(new short[]{
                backend.cpu.kernels.CpuDTypeOps.toBFloat16Bits(1.0f),
                backend.cpu.kernels.CpuDTypeOps.toBFloat16Bits(2.0f)
        }, new int[]{2}, null, "f16", DataType.BFLOAT16);
        Tensor f32 = new Tensor(new float[]{10f, 20f}, new int[]{2}, null, "f32", DataType.FLOAT32);
        Tensor f64 = new Tensor(new double[]{100, 200}, new int[]{2}, null, "f64", DataType.FLOAT64);

        Tensor out16_32 = Tensor.where(condition, f16, f32);
        CompiledGraph.compile(out16_32, OptimizerConfig.noOptimization()).execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
        assertEquals(DataType.FLOAT32, out16_32.getDataType());
        assertArrayEquals(new double[]{1, 20}, out16_32.toDoubleArrayCopy(), 1e-6);

        Tensor out32_64 = Tensor.where(condition, f32, f64);
        CompiledGraph.compile(out32_64, OptimizerConfig.noOptimization()).execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
        assertEquals(DataType.FLOAT64, out32_64.getDataType());
        assertArrayEquals(new double[]{10, 200}, out32_64.toDoubleArrayCopy(), 1e-9);

        Tensor out16_64 = Tensor.where(condition, f16, f64);
        CompiledGraph.compile(out16_64, OptimizerConfig.noOptimization()).execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
        assertEquals(DataType.FLOAT64, out16_64.getDataType());
        assertArrayEquals(new double[]{1, 200}, out16_64.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void whereSupportsNonContiguousConditionView() {
        Tensor condition = new Tensor(new byte[]{1, 0, 1, 0}, new int[]{2, 2}, new int[]{1, 2}, null, "cond_view", DataType.BOOL);
        Tensor ifTrue = new Tensor(new double[]{1, 2, 3, 4}, new int[]{2, 2}, null, "x", DataType.FLOAT64);
        Tensor ifFalse = new Tensor(new double[]{10, 20, 30, 40}, new int[]{2, 2}, null, "y", DataType.FLOAT64);

        Tensor out = Tensor.where(condition, ifTrue, ifFalse);
        CompiledGraph.compile(out, OptimizerConfig.noOptimization()).execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{1, 2, 30, 40}, out.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void whereSupportsNonContiguousBranchView() {
        Tensor condition = new Tensor(new byte[]{1, 0, 1, 0}, new int[]{2, 2}, null, "cond", DataType.BOOL);
        Tensor ifTrue = new Tensor(new double[]{1, 2, 3, 4}, new int[]{2, 2}, new int[]{1, 2}, null, "x_view", DataType.FLOAT64);
        Tensor ifFalse = new Tensor(new double[]{10, 20, 30, 40}, new int[]{2, 2}, null, "y", DataType.FLOAT64);

        Tensor out = Tensor.where(condition, ifTrue, ifFalse);
        CompiledGraph.compile(out, OptimizerConfig.noOptimization()).execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{1, 20, 2, 40}, out.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void bfloat16WhereBroadcastChainSupportsContinuationBackedExecution() {
        Tensor condition = new Tensor(new byte[]{1, 0}, new int[]{2, 1}, null, "cond", DataType.BOOL);
        Tensor base = new Tensor(new double[]{1, 2, 3, 4}, new int[]{2, 2}, null, "base", DataType.BFLOAT16);
        Tensor ifTrue = base.add(Tensor.scalar(1.0, DataType.BFLOAT16));
        Tensor ifFalse = Tensor.scalar(-10.0, DataType.BFLOAT16);
        Tensor out = Tensor.where(condition, ifTrue, ifFalse).relu();

        CompiledGraph.compile(out, OptimizerConfig.noOptimization()).execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertEquals(DataType.BFLOAT16, out.getDataType());
        assertArrayEquals(new double[]{2.0, 3.0, 0.0, 0.0}, out.toDoubleArrayCopy(), 1e-2);
    }

    @Test
    void whereRejectsNonBoolCondition() {
        Tensor condition = new Tensor(new double[]{1, 0}, new int[]{2}, null, "cond", DataType.FLOAT64);
        Tensor ifTrue = new Tensor(new double[]{1, 2}, new int[]{2}, null, "x", DataType.FLOAT64);
        Tensor ifFalse = new Tensor(new double[]{10, 20}, new int[]{2}, null, "y", DataType.FLOAT64);

        assertThrows(IllegalArgumentException.class, () -> Tensor.where(condition, ifTrue, ifFalse));
    }

    @Test
    void whereRejectsBoolBranches() {
        Tensor condition = new Tensor(new byte[]{1, 0}, new int[]{2}, null, "cond", DataType.BOOL);
        Tensor boolBranch = new Tensor(new byte[]{1, 1}, new int[]{2}, null, "mask", DataType.BOOL);
        Tensor numericBranch = new Tensor(new double[]{10, 20}, new int[]{2}, null, "y", DataType.FLOAT64);

        assertThrows(IllegalArgumentException.class, () -> Tensor.where(condition, boolBranch, numericBranch));
        assertThrows(IllegalArgumentException.class, () -> Tensor.where(condition, numericBranch, boolBranch));
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

    @Test
    void minimumAndMaximumForwardMatchExpectedValues() {
        Tensor a = new Tensor(new double[]{1, 5, 3}, new int[]{3}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{2, 4, 3}, new int[]{3}, null, "b", DataType.FLOAT64);

        Tensor min = a.minimum(b);
        Tensor max = a.maximum(b);

        CompiledGraph.compile(min, OptimizerConfig.noOptimization()).execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
        CompiledGraph.compile(max, OptimizerConfig.noOptimization()).execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{1, 4, 3}, min.toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{2, 5, 3}, max.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void minimumAndMaximumUseWhereStyleTieGradientSemantics() {
        Tensor aMin = new Tensor(new double[]{1, 5, 3}, new int[]{3}, null, "aMin", DataType.FLOAT64);
        Tensor bMin = new Tensor(new double[]{2, 4, 3}, new int[]{3}, null, "bMin", DataType.FLOAT64);
        aMin.setRequiresGrad(true);
        bMin.setRequiresGrad(true);

        Tensor min = aMin.minimum(bMin);
        CompiledGraph.compile(min, OptimizerConfig.trainingDefaults()).execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);
        assertArrayEquals(new double[]{1, 0, 0}, aMin.getGradient().toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{0, 1, 1}, bMin.getGradient().toDoubleArrayCopy(), 1e-9);

        Tensor aMax = new Tensor(new double[]{1, 5, 3}, new int[]{3}, null, "aMax", DataType.FLOAT64);
        Tensor bMax = new Tensor(new double[]{2, 4, 3}, new int[]{3}, null, "bMax", DataType.FLOAT64);
        aMax.setRequiresGrad(true);
        bMax.setRequiresGrad(true);

        Tensor max = aMax.maximum(bMax);
        CompiledGraph.compile(max, OptimizerConfig.trainingDefaults()).execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);
        assertArrayEquals(new double[]{0, 1, 0}, aMax.getGradient().toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{1, 0, 1}, bMax.getGradient().toDoubleArrayCopy(), 1e-9);
    }
}
