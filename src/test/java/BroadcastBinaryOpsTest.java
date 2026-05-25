import backend.runtime.ExecutionMode;
import config.compile.CompileConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertNotNull;

public class BroadcastBinaryOpsTest {
    @Test
    public void testSubBroadcastForwardAndBackward() {
        Tensor a = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{10, 20, 30}, new int[]{3}, null, "b", DataType.FLOAT64);
        a.setRequiresGrad(true);
        b.setRequiresGrad(true);

        Tensor out = a.sub(b);
        CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.trainingDefaults()).execute(ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(new double[]{-9, -18, -27, -6, -15, -24}, out.toDoubleArrayCopy(), 1e-9);
        assertNotNull(a.getGradient());
        assertNotNull(b.getGradient());
        assertArrayEquals(new double[]{1, 1, 1, 1, 1, 1}, a.getGradient().toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{-2, -2, -2}, b.getGradient().toDoubleArrayCopy(), 1e-9);
    }

    @Test
    public void testMulBroadcastBackwardReduction() {
        Tensor a = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{10, 20, 30}, new int[]{3}, null, "b", DataType.FLOAT64);
        a.setRequiresGrad(true);
        b.setRequiresGrad(true);

        Tensor out = a.mul(b);
        CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.trainingDefaults()).execute(ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(new double[]{10, 40, 90, 40, 100, 180}, out.toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{10, 20, 30, 10, 20, 30}, a.getGradient().toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{5, 7, 9}, b.getGradient().toDoubleArrayCopy(), 1e-9);
    }

    @Test
    public void testAddBroadcastBackwardToScalar() {
        Tensor a = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{10}, new int[]{1}, null, "b_scalar", DataType.FLOAT64);
        a.setRequiresGrad(true);
        b.setRequiresGrad(true);

        Tensor out = a.add(b);
        CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.trainingDefaults()).execute(ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(new double[]{11, 12, 13, 14, 15, 16}, out.toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{1, 1, 1, 1, 1, 1}, a.getGradient().toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{6}, b.getGradient().toDoubleArrayCopy(), 1e-9);
    }

    @Test
    public void testAddBroadcastBackwardRankMismatch() {
        Tensor a = new Tensor(
                new double[]{1, 2, 3, 4, 5, 6, 7, 8},
                new int[]{2, 1, 4},
                null,
                "a",
                DataType.FLOAT64
        );
        Tensor b = new Tensor(
                new double[]{10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120},
                new int[]{1, 3, 4},
                null,
                "b",
                DataType.FLOAT64
        );
        a.setRequiresGrad(true);
        b.setRequiresGrad(true);

        Tensor out = a.add(b); // out shape [2,3,4]
        CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.trainingDefaults()).execute(ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(new int[]{2, 3, 4}, out.getShape());
        assertArrayEquals(new int[]{2, 1, 4}, a.getGradient().getShape());
        assertArrayEquals(new int[]{1, 3, 4}, b.getGradient().getShape());
        assertArrayEquals(new double[]{3, 3, 3, 3, 3, 3, 3, 3}, a.getGradient().toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2}, b.getGradient().toDoubleArrayCopy(), 1e-9);
    }

    @Test
    public void testBroadcastRankMismatchAlignsFromRight() {
        Tensor a = new Tensor(new double[]{
                1, 2, 3, 4,
                5, 6, 7, 8
        }, new int[]{2, 1, 4}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{
                10, 20, 30, 40,
                50, 60, 70, 80,
                90, 100, 110, 120
        }, new int[]{3, 4}, null, "b", DataType.FLOAT64);

        Tensor out = a.add(b);
        CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        assertArrayEquals(new int[]{2, 3, 4}, out.getShape());
        assertArrayEquals(new double[]{
                11, 22, 33, 44,
                51, 62, 73, 84,
                91, 102, 113, 124,
                15, 26, 37, 48,
                55, 66, 77, 88,
                95, 106, 117, 128
        }, out.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    public void testBroadcastRankMismatchAlignsFromRightWhenLeftOperandHasLowerRank() {
        Tensor a = new Tensor(new double[]{
                10, 20, 30, 40,
                50, 60, 70, 80,
                90, 100, 110, 120
        }, new int[]{3, 4}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{
                1, 2, 3, 4,
                5, 6, 7, 8
        }, new int[]{2, 1, 4}, null, "b", DataType.FLOAT64);

        Tensor out = a.add(b);
        CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        assertArrayEquals(new int[]{2, 3, 4}, out.getShape());
        assertArrayEquals(new double[]{
                11, 22, 33, 44,
                51, 62, 73, 84,
                91, 102, 113, 124,
                15, 26, 37, 48,
                55, 66, 77, 88,
                95, 106, 117, 128
        }, out.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    public void testBroadcastSupportsLeadingSingletonExpansionAcrossFourDimensions() {
        Tensor a = new Tensor(new double[]{
                1, 2, 3, 4,
                5, 6, 7, 8
        }, new int[]{1, 1, 2, 4}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{
                10, 20, 30, 40,
                50, 60, 70, 80,
                90, 100, 110, 120,
                130, 140, 150, 160,
                170, 180, 190, 200,
                210, 220, 230, 240
        }, new int[]{2, 3, 1, 4}, null, "b", DataType.FLOAT64);

        Tensor out = a.add(b);
        CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        assertArrayEquals(new int[]{2, 3, 2, 4}, out.getShape());
        assertArrayEquals(new double[]{
                11, 22, 33, 44, 15, 26, 37, 48,
                51, 62, 73, 84, 55, 66, 77, 88,
                91, 102, 113, 124, 95, 106, 117, 128,
                131, 142, 153, 164, 135, 146, 157, 168,
                171, 182, 193, 204, 175, 186, 197, 208,
                211, 222, 233, 244, 215, 226, 237, 248
        }, out.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    public void testBroadcastRejectsIncompatibleShapes() {
        Tensor a = new Tensor(new double[2 * 3 * 4], new int[]{2, 3, 4}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[2 * 4], new int[]{2, 4}, null, "b", DataType.FLOAT64);
        Tensor c = new Tensor(new double[1 * 2 * 4], new int[]{1, 2, 4}, null, "c", DataType.FLOAT64);

        assertThrows(IllegalArgumentException.class, () -> a.add(b));
        assertThrows(IllegalArgumentException.class, () -> a.sub(b));
        assertThrows(IllegalArgumentException.class, () -> a.mul(b));
        assertThrows(IllegalArgumentException.class, () -> a.div(b));
        assertThrows(IllegalArgumentException.class, () -> a.min(b));
        assertThrows(IllegalArgumentException.class, () -> a.max(b));

        assertThrows(IllegalArgumentException.class, () -> a.add(c));
        assertThrows(IllegalArgumentException.class, () -> a.sub(c));
        assertThrows(IllegalArgumentException.class, () -> a.mul(c));
        assertThrows(IllegalArgumentException.class, () -> a.div(c));
        assertThrows(IllegalArgumentException.class, () -> a.min(c));
        assertThrows(IllegalArgumentException.class, () -> a.max(c));
    }

    @Test
    public void testDivBroadcastBackwardReduction() {
        Tensor a = new Tensor(new double[]{2, 4, 6, 8, 10, 12}, new int[]{2, 3}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{2, 2, 3}, new int[]{3}, null, "b", DataType.FLOAT64);
        a.setRequiresGrad(true);
        b.setRequiresGrad(true);

        Tensor out = a.div(b);
        CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.trainingDefaults()).execute(ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(new double[]{1, 2, 2, 4, 5, 4}, out.toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{0.5, 0.5, 1.0 / 3.0, 0.5, 0.5, 1.0 / 3.0}, a.getGradient().toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{-2.5, -3.5, -2.0}, b.getGradient().toDoubleArrayCopy(), 1e-9);
    }

    @Test
    public void testMinAndMaxBroadcastBackward() {
        Tensor a = new Tensor(new double[]{1, 5, 3, 7, 2, 9}, new int[]{2, 3}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{4, 4, 4}, new int[]{3}, null, "b", DataType.FLOAT64);
        a.setRequiresGrad(true);
        b.setRequiresGrad(true);

        Tensor minOut = a.min(b);
        CompiledGraph.compile(minOut, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.trainingDefaults()).execute(ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(new double[]{1, 4, 3, 4, 2, 4}, minOut.toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{1, 0, 1, 0, 1, 0}, a.getGradient().toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{1, 1, 1}, b.getGradient().toDoubleArrayCopy(), 1e-9);

        TensorInternalAccess.clearGradient(a);
        TensorInternalAccess.clearGradient(b);

        Tensor maxOut = a.max(b);
        CompiledGraph.compile(maxOut, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.trainingDefaults()).execute(ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(new double[]{4, 5, 4, 7, 4, 9}, maxOut.toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{0, 1, 0, 1, 0, 1}, a.getGradient().toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{1, 1, 1}, b.getGradient().toDoubleArrayCopy(), 1e-9);
    }

    @Test
    public void testMinAndMaxBroadcastBackwardFloat32() {
        Tensor a = new Tensor(new float[]{1f, 5f, 3f, 7f, 2f, 9f}, new int[]{2, 3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{4f, 4f, 4f}, new int[]{3}, null, "b", DataType.FLOAT32);
        a.setRequiresGrad(true);
        b.setRequiresGrad(true);

        Tensor minOut = a.min(b);
        CompiledGraph.compile(minOut, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.trainingDefaults()).execute(ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(new double[]{1, 4, 3, 4, 2, 4}, minOut.toDoubleArrayCopy(), 1e-6);
        assertArrayEquals(new double[]{1, 0, 1, 0, 1, 0}, a.getGradient().toDoubleArrayCopy(), 1e-6);
        assertArrayEquals(new double[]{1, 1, 1}, b.getGradient().toDoubleArrayCopy(), 1e-6);

        TensorInternalAccess.clearGradient(a);
        TensorInternalAccess.clearGradient(b);

        Tensor maxOut = a.max(b);
        CompiledGraph.compile(maxOut, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.trainingDefaults()).execute(ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(new double[]{4, 5, 4, 7, 4, 9}, maxOut.toDoubleArrayCopy(), 1e-6);
        assertArrayEquals(new double[]{0, 1, 0, 1, 0, 1}, a.getGradient().toDoubleArrayCopy(), 1e-6);
        assertArrayEquals(new double[]{1, 1, 1}, b.getGradient().toDoubleArrayCopy(), 1e-6);
    }

    @Test
    public void testMinAndMaxBroadcastBackwardBFloat16() {
        Tensor a = new Tensor(new short[]{
                tensor.dtype.TensorDTypeOps.toBFloat16Bits(1f), tensor.dtype.TensorDTypeOps.toBFloat16Bits(5f), tensor.dtype.TensorDTypeOps.toBFloat16Bits(3f),
                tensor.dtype.TensorDTypeOps.toBFloat16Bits(7f), tensor.dtype.TensorDTypeOps.toBFloat16Bits(2f), tensor.dtype.TensorDTypeOps.toBFloat16Bits(9f)
        }, new int[]{2, 3}, null, "a", DataType.BFLOAT16);
        Tensor b = new Tensor(new short[]{
                tensor.dtype.TensorDTypeOps.toBFloat16Bits(4f), tensor.dtype.TensorDTypeOps.toBFloat16Bits(4f), tensor.dtype.TensorDTypeOps.toBFloat16Bits(4f)
        }, new int[]{3}, null, "b", DataType.BFLOAT16);
        a.setRequiresGrad(true);
        b.setRequiresGrad(true);

        Tensor minOut = a.min(b);
        CompiledGraph.compile(minOut, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.trainingDefaults()).execute(ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(new double[]{1, 4, 3, 4, 2, 4}, minOut.toDoubleArrayCopy(), 1e-2);
        assertArrayEquals(new double[]{1, 0, 1, 0, 1, 0}, a.getGradient().toDoubleArrayCopy(), 1e-2);
        assertArrayEquals(new double[]{1, 1, 1}, b.getGradient().toDoubleArrayCopy(), 1e-2);

        TensorInternalAccess.clearGradient(a);
        TensorInternalAccess.clearGradient(b);

        Tensor maxOut = a.max(b);
        CompiledGraph.compile(maxOut, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.trainingDefaults()).execute(ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(new double[]{4, 5, 4, 7, 4, 9}, maxOut.toDoubleArrayCopy(), 1e-2);
        assertArrayEquals(new double[]{0, 1, 0, 1, 0, 1}, a.getGradient().toDoubleArrayCopy(), 1e-2);
        assertArrayEquals(new double[]{1, 1, 1}, b.getGradient().toDoubleArrayCopy(), 1e-2);
    }

    @Test
    public void testMinMaxTieSplitsGradientEvenlyScalarBroadcast() {
        Tensor a = new Tensor(new double[]{4, 4, 4, 4}, new int[]{2, 2}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{4}, new int[]{1}, null, "b", DataType.FLOAT64);
        a.setRequiresGrad(true);
        b.setRequiresGrad(true);

        Tensor minOut = a.min(b);
        CompiledGraph.compile(minOut, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.trainingDefaults()).execute(ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(new double[]{0.5, 0.5, 0.5, 0.5}, a.getGradient().toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{2.0}, b.getGradient().toDoubleArrayCopy(), 1e-9);

        TensorInternalAccess.clearGradient(a);
        TensorInternalAccess.clearGradient(b);

        Tensor maxOut = a.max(b);
        CompiledGraph.compile(maxOut, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.trainingDefaults()).execute(ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(new double[]{0.5, 0.5, 0.5, 0.5}, a.getGradient().toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{2.0}, b.getGradient().toDoubleArrayCopy(), 1e-9);
    }
}
