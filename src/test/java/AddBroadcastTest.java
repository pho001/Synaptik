import backend.kernels.cpu.CpuDTypeOps;
import backend.runtime.ExecutionMode;
import config.optimizer.OptimizerConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
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
        CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

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
        CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);

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
        CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

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
        CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);

        assertNotNull(a.getGradient());
        assertNotNull(b.getGradient());
        assertArrayEquals(new double[]{1, 1, 1, 1, 1, 1}, a.getGradient().toDoubleArrayCopy(), 1e-6);
        assertArrayEquals(new double[]{2, 2, 2}, b.getGradient().toDoubleArrayCopy(), 1e-6);
    }

    @Test
    public void testAddBackwardBroadcastBFloat16() {
        Tensor a = new Tensor(new short[]{
                CpuDTypeOps.toBFloat16Bits(1f), CpuDTypeOps.toBFloat16Bits(2f), CpuDTypeOps.toBFloat16Bits(3f),
                CpuDTypeOps.toBFloat16Bits(4f), CpuDTypeOps.toBFloat16Bits(5f), CpuDTypeOps.toBFloat16Bits(6f)
        }, new int[]{2, 3}, null, "a", DataType.BFLOAT16);
        Tensor b = new Tensor(new short[]{
                CpuDTypeOps.toBFloat16Bits(10f), CpuDTypeOps.toBFloat16Bits(20f), CpuDTypeOps.toBFloat16Bits(30f)
        }, new int[]{3}, null, "b", DataType.BFLOAT16);
        a.setRequiresGrad(true);
        b.setRequiresGrad(true);

        Tensor out = a.add(b);
        CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);

        assertNotNull(a.getGradient());
        assertNotNull(b.getGradient());
        assertArrayEquals(new double[]{1, 1, 1, 1, 1, 1}, a.getGradient().toDoubleArrayCopy(), 1e-2);
        assertArrayEquals(new double[]{2, 2, 2}, b.getGradient().toDoubleArrayCopy(), 1e-2);
    }
}
