import graph.optimizer.GraphOptimizer;
import tensor.DataType;
import tensor.Tensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

public class BroadcastBinaryOpsTest {
    @Test
    public void testSubBroadcastForwardAndBackward() {
        Tensor a = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{10, 20, 30}, new int[]{3}, null, "b", DataType.FLOAT64);
        a.setRequiresGrad(true);
        b.setRequiresGrad(true);

        Tensor out = a.sub(b);
        TestGraphSupport.execute(out, new GraphOptimizer(), config.runtime.RuntimeConfig.trainingDefaults(), backend.runtime.ExecutionMode.FORWARD_BACKWARD);

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
        TestGraphSupport.execute(out, new GraphOptimizer(), config.runtime.RuntimeConfig.trainingDefaults(), backend.runtime.ExecutionMode.FORWARD_BACKWARD);

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
        TestGraphSupport.execute(out, new GraphOptimizer(), config.runtime.RuntimeConfig.trainingDefaults(), backend.runtime.ExecutionMode.FORWARD_BACKWARD);

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
        TestGraphSupport.execute(out, new GraphOptimizer(), config.runtime.RuntimeConfig.trainingDefaults(), backend.runtime.ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(new int[]{2, 3, 4}, out.getShape());
        assertArrayEquals(new int[]{2, 1, 4}, a.getGradient().getShape());
        assertArrayEquals(new int[]{1, 3, 4}, b.getGradient().getShape());
        assertArrayEquals(new double[]{3, 3, 3, 3, 3, 3, 3, 3}, a.getGradient().toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2}, b.getGradient().toDoubleArrayCopy(), 1e-9);
    }

    @Test
    public void testDivBroadcastBackwardReduction() {
        Tensor a = new Tensor(new double[]{2, 4, 6, 8, 10, 12}, new int[]{2, 3}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{2, 2, 3}, new int[]{3}, null, "b", DataType.FLOAT64);
        a.setRequiresGrad(true);
        b.setRequiresGrad(true);

        Tensor out = a.div(b);
        TestGraphSupport.execute(out, new GraphOptimizer(), config.runtime.RuntimeConfig.trainingDefaults(), backend.runtime.ExecutionMode.FORWARD_BACKWARD);

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
        TestGraphSupport.execute(minOut, new GraphOptimizer(), config.runtime.RuntimeConfig.trainingDefaults(), backend.runtime.ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(new double[]{1, 4, 3, 4, 2, 4}, minOut.toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{1, 0, 1, 0, 1, 0}, a.getGradient().toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{1, 1, 1}, b.getGradient().toDoubleArrayCopy(), 1e-9);

        a.setGradient(null);
        b.setGradient(null);

        Tensor maxOut = a.max(b);
        TestGraphSupport.execute(maxOut, new GraphOptimizer(), config.runtime.RuntimeConfig.trainingDefaults(), backend.runtime.ExecutionMode.FORWARD_BACKWARD);

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
        TestGraphSupport.execute(minOut, new GraphOptimizer(), config.runtime.RuntimeConfig.trainingDefaults(), backend.runtime.ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(new double[]{1, 4, 3, 4, 2, 4}, minOut.toDoubleArrayCopy(), 1e-6);
        assertArrayEquals(new double[]{1, 0, 1, 0, 1, 0}, a.getGradient().toDoubleArrayCopy(), 1e-6);
        assertArrayEquals(new double[]{1, 1, 1}, b.getGradient().toDoubleArrayCopy(), 1e-6);

        a.setGradient(null);
        b.setGradient(null);

        Tensor maxOut = a.max(b);
        TestGraphSupport.execute(maxOut, new GraphOptimizer(), config.runtime.RuntimeConfig.trainingDefaults(), backend.runtime.ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(new double[]{4, 5, 4, 7, 4, 9}, maxOut.toDoubleArrayCopy(), 1e-6);
        assertArrayEquals(new double[]{0, 1, 0, 1, 0, 1}, a.getGradient().toDoubleArrayCopy(), 1e-6);
        assertArrayEquals(new double[]{1, 1, 1}, b.getGradient().toDoubleArrayCopy(), 1e-6);
    }

    @Test
    public void testMinAndMaxBroadcastBackwardFloat16() {
        Tensor a = new Tensor(new short[]{
                backend.kernels.cpu.CpuDTypeOps.toHalfBits(1f), backend.kernels.cpu.CpuDTypeOps.toHalfBits(5f), backend.kernels.cpu.CpuDTypeOps.toHalfBits(3f),
                backend.kernels.cpu.CpuDTypeOps.toHalfBits(7f), backend.kernels.cpu.CpuDTypeOps.toHalfBits(2f), backend.kernels.cpu.CpuDTypeOps.toHalfBits(9f)
        }, new int[]{2, 3}, null, "a", DataType.FLOAT16);
        Tensor b = new Tensor(new short[]{
                backend.kernels.cpu.CpuDTypeOps.toHalfBits(4f), backend.kernels.cpu.CpuDTypeOps.toHalfBits(4f), backend.kernels.cpu.CpuDTypeOps.toHalfBits(4f)
        }, new int[]{3}, null, "b", DataType.FLOAT16);
        a.setRequiresGrad(true);
        b.setRequiresGrad(true);

        Tensor minOut = a.min(b);
        TestGraphSupport.execute(minOut, new GraphOptimizer(), config.runtime.RuntimeConfig.trainingDefaults(), backend.runtime.ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(new double[]{1, 4, 3, 4, 2, 4}, minOut.toDoubleArrayCopy(), 1e-2);
        assertArrayEquals(new double[]{1, 0, 1, 0, 1, 0}, a.getGradient().toDoubleArrayCopy(), 1e-2);
        assertArrayEquals(new double[]{1, 1, 1}, b.getGradient().toDoubleArrayCopy(), 1e-2);

        a.setGradient(null);
        b.setGradient(null);

        Tensor maxOut = a.max(b);
        TestGraphSupport.execute(maxOut, new GraphOptimizer(), config.runtime.RuntimeConfig.trainingDefaults(), backend.runtime.ExecutionMode.FORWARD_BACKWARD);

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
        TestGraphSupport.execute(minOut, new GraphOptimizer(), config.runtime.RuntimeConfig.trainingDefaults(), backend.runtime.ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(new double[]{0.5, 0.5, 0.5, 0.5}, a.getGradient().toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{2.0}, b.getGradient().toDoubleArrayCopy(), 1e-9);

        a.setGradient(null);
        b.setGradient(null);

        Tensor maxOut = a.max(b);
        TestGraphSupport.execute(maxOut, new GraphOptimizer(), config.runtime.RuntimeConfig.trainingDefaults(), backend.runtime.ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(new double[]{0.5, 0.5, 0.5, 0.5}, a.getGradient().toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{2.0}, b.getGradient().toDoubleArrayCopy(), 1e-9);
    }
}
