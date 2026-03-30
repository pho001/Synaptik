import backend.ComputeEngine;
import config.backend.CpuKernelConfig;
import graph.optimizer.GraphOptimizer;
import tensor.DataType;
import tensor.Tensor;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class MatMulTest {
    @AfterEach
    void resetCpuConfig() {
        ComputeEngine.setCpuKernelConfig(CpuKernelConfig.defaultsTraining());
    }

    @Test
    void matMulForwardAndBackwardFloat64() {
        Tensor a = new Tensor(new double[]{1, 2, 3, 4}, new int[]{2, 2}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{5, 6, 7, 8}, new int[]{2, 2}, null, "b", DataType.FLOAT64);
        a.setRequiresGrad(true);
        b.setRequiresGrad(true);

        Tensor c = a.matmul(b);
        c.compute(new GraphOptimizer());
        assertArrayEquals(new double[]{19, 22, 43, 50}, c.toDoubleArrayCopy(), 1e-9);

        c.getCompiledGraph().setTrainingModeOn();
        c.compute(new GraphOptimizer());
        assertArrayEquals(new double[]{11, 15, 11, 15}, a.getGradient().toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{4, 4, 6, 6}, b.getGradient().toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void matMulForwardFloat32TiledParallel() {
        ComputeEngine.setCpuKernelConfig(new CpuKernelConfig(
                4, 1, 2, 1,
                1, 1,
                0, 2, 64
        ));

        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{7f, 8f, 9f, 10f, 11f, 12f}, new int[]{3, 2}, null, "b", DataType.FLOAT32);
        Tensor c = a.matmul(b);
        c.compute(new GraphOptimizer());

        assertArrayEquals(new double[]{58, 64, 139, 154}, c.toDoubleArrayCopy(), 1e-5);
    }

    @Test
    void matMulShapeMismatchThrows() {
        Tensor a = new Tensor(new double[]{1, 2, 3, 4}, new int[]{2, 2}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{3, 2}, null, "b", DataType.FLOAT64);
        assertThrows(IllegalArgumentException.class, () -> a.matmul(b));
    }
}

