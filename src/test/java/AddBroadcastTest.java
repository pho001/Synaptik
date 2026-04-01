import graph.optimizer.GraphOptimizer;
import backend.kernels.cpu.CpuDTypeOps;
import tensor.DataType;
import tensor.Tensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

public class AddBroadcastTest {
    @Test
    public void testAddForwardBroadcastFloat64() {
        Tensor a = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{10, 20, 30}, new int[]{3}, null, "b", DataType.FLOAT64);

        Tensor out = a.add(b);
        TestGraphSupport.execute(out, new GraphOptimizer());

        assertArrayEquals(new double[]{11, 22, 33, 14, 25, 36}, out.toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new int[]{2, 3}, out.getShape());
    }

    @Test
    public void testAddBackwardBroadcastFloat64() {
        Tensor a = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{10, 20, 30}, new int[]{3}, null, "b", DataType.FLOAT64);
        a.setRequiresGrad(true);
        b.setRequiresGrad(true);

        Tensor out = a.add(b);
        TestGraphSupport.execute(out, new GraphOptimizer(), config.runtime.RuntimeConfig.trainingDefaults(), backend.runtime.ExecutionMode.FORWARD_BACKWARD);

        assertNotNull(a.getGradient());
        assertNotNull(b.getGradient());
        assertArrayEquals(new double[]{1, 1, 1, 1, 1, 1}, a.getGradient().toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{2, 2, 2}, b.getGradient().toDoubleArrayCopy(), 1e-9);
    }

    @Test
    public void testAddForwardBroadcastFloat32() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{10f, 20f, 30f}, new int[]{3}, null, "b", DataType.FLOAT32);

        Tensor out = a.add(b);
        TestGraphSupport.execute(out, new GraphOptimizer());

        assertArrayEquals(new double[]{11, 22, 33, 14, 25, 36}, out.toDoubleArrayCopy(), 1e-6);
        assertArrayEquals(new int[]{2, 3}, out.getShape());
    }

    @Test
    public void testAddBackwardBroadcastFloat32() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{10f, 20f, 30f}, new int[]{3}, null, "b", DataType.FLOAT32);
        a.setRequiresGrad(true);
        b.setRequiresGrad(true);

        Tensor out = a.add(b);
        TestGraphSupport.execute(out, new GraphOptimizer(), config.runtime.RuntimeConfig.trainingDefaults(), backend.runtime.ExecutionMode.FORWARD_BACKWARD);

        assertNotNull(a.getGradient());
        assertNotNull(b.getGradient());
        assertArrayEquals(new double[]{1, 1, 1, 1, 1, 1}, a.getGradient().toDoubleArrayCopy(), 1e-6);
        assertArrayEquals(new double[]{2, 2, 2}, b.getGradient().toDoubleArrayCopy(), 1e-6);
    }

    @Test
    public void testAddBackwardBroadcastFloat16() {
        Tensor a = new Tensor(new short[]{
                CpuDTypeOps.toHalfBits(1f), CpuDTypeOps.toHalfBits(2f), CpuDTypeOps.toHalfBits(3f),
                CpuDTypeOps.toHalfBits(4f), CpuDTypeOps.toHalfBits(5f), CpuDTypeOps.toHalfBits(6f)
        }, new int[]{2, 3}, null, "a", DataType.FLOAT16);
        Tensor b = new Tensor(new short[]{
                CpuDTypeOps.toHalfBits(10f), CpuDTypeOps.toHalfBits(20f), CpuDTypeOps.toHalfBits(30f)
        }, new int[]{3}, null, "b", DataType.FLOAT16);
        a.setRequiresGrad(true);
        b.setRequiresGrad(true);

        Tensor out = a.add(b);
        TestGraphSupport.execute(out, new GraphOptimizer(), config.runtime.RuntimeConfig.trainingDefaults(), backend.runtime.ExecutionMode.FORWARD_BACKWARD);

        assertNotNull(a.getGradient());
        assertNotNull(b.getGradient());
        assertArrayEquals(new double[]{1, 1, 1, 1, 1, 1}, a.getGradient().toDoubleArrayCopy(), 1e-2);
        assertArrayEquals(new double[]{2, 2, 2}, b.getGradient().toDoubleArrayCopy(), 1e-2);
    }
}
