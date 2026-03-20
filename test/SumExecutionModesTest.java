import Backend.ComputeEngine;
import Config.backend.CpuKernelConfig;
import Config.backend.SumAccuracyMode;
import Graph.optimizer.GraphOptimizer;
import Tensor.Tensor;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class SumExecutionModesTest {
    private static final double EPS = 1e-9;

    @AfterEach
    public void resetCpuConfig() {
        ComputeEngine.setCpuKernelConfig(CpuKernelConfig.defaultsTraining());
    }

    @Test
    public void testSumAllParallelVectorMatchesReference() {
        ComputeEngine.setCpuKernelConfig(new CpuKernelConfig(
                4, 32, 32, 32,
                1, 1, 0, 8, 256, 1_000_000_000, SumAccuracyMode.FAST
        ));

        double[] values = new double[20_000];
        for (int i = 0; i < values.length; i++) {
            values[i] = (i % 17) * 0.25 - 1.75;
        }

        Tensor a = new Tensor(values, new int[]{values.length}, null, "a");
        Tensor s = a.sum();
        s.compute(new GraphOptimizer());

        assertEquals(referenceSumContiguous(values), s.getData()[0], EPS);
    }

    @Test
    public void testSumAxisLastDimParallelVectorMatchesReference() {
        ComputeEngine.setCpuKernelConfig(new CpuKernelConfig(
                4, 32, 32, 32,
                1, 1, 0, 8, 128, 1_000_000_000, SumAccuracyMode.FAST
        ));

        int rows = 128;
        int cols = 256;
        double[] values = new double[rows * cols];
        for (int i = 0; i < values.length; i++) {
            values[i] = ((i % 31) - 15) * 0.5;
        }

        Tensor a = new Tensor(values, new int[]{rows, cols}, null, "matrix");
        Tensor s = a.sum(1);
        s.compute(new GraphOptimizer());

        double[] expected = new double[rows];
        for (int r = 0; r < rows; r++) {
            double acc = 0.0;
            int base = r * cols;
            for (int c = 0; c < cols; c++) {
                acc += values[base + c];
            }
            expected[r] = acc;
        }
        assertArrayEquals(expected, s.getData(), EPS);
    }

    @Test
    public void testSumNonContiguousStridedVsMaterializedEquivalence() {
        Tensor a = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, new int[]{1, 2}, null, "a_noncontig");

        ComputeEngine.setCpuKernelConfig(new CpuKernelConfig(
                4, 32, 32, 32,
                1_024, 100_000, 0, 4, 4_096, 1_000_000_000, SumAccuracyMode.FAST
        ));
        Tensor stridedAll = a.sum();
        stridedAll.compute(new GraphOptimizer());
        Tensor stridedAxis = a.sum(1);
        stridedAxis.compute(new GraphOptimizer());

        ComputeEngine.setCpuKernelConfig(new CpuKernelConfig(
                4, 32, 32, 32,
                1_024, 100_000, 0, 4, 4_096, 0, SumAccuracyMode.FAST
        ));
        Tensor materializedAll = a.sum();
        materializedAll.compute(new GraphOptimizer());
        Tensor materializedAxis = a.sum(1);
        materializedAxis.compute(new GraphOptimizer());

        assertArrayEquals(stridedAll.getData(), materializedAll.getData(), EPS);
        assertArrayEquals(stridedAxis.getData(), materializedAxis.getData(), EPS);
        assertArrayEquals(new double[]{9.0, 12.0}, stridedAxis.getData(), EPS);
    }

    @Test
    public void testSumAllAccuracyModesStayClose() {
        double[] values = new double[50_000];
        for (int i = 0; i < values.length; i++) {
            values[i] = (i % 2 == 0 ? 1.0 : -1.0) * (1e-6 * (i % 23));
        }

        Tensor fastTensor = new Tensor(values.clone(), new int[]{values.length}, null, "fast");
        ComputeEngine.setCpuKernelConfig(new CpuKernelConfig(
                4, 32, 32, 32,
                1, 1, 0, 8, 512, 1_000_000_000, SumAccuracyMode.FAST
        ));
        Tensor fast = fastTensor.sum();
        fast.compute(new GraphOptimizer());

        Tensor kahanTensor = new Tensor(values.clone(), new int[]{values.length}, null, "kahan");
        ComputeEngine.setCpuKernelConfig(new CpuKernelConfig(
                4, 32, 32, 32,
                1, 1, 0, 8, 512, 1_000_000_000, SumAccuracyMode.KAHAN
        ));
        Tensor kahan = kahanTensor.sum();
        kahan.compute(new GraphOptimizer());

        Tensor neumaierTensor = new Tensor(values.clone(), new int[]{values.length}, null, "neumaier");
        ComputeEngine.setCpuKernelConfig(new CpuKernelConfig(
                4, 32, 32, 32,
                1, 1, 0, 8, 512, 1_000_000_000, SumAccuracyMode.NEUMAIER
        ));
        Tensor neumaier = neumaierTensor.sum();
        neumaier.compute(new GraphOptimizer());

        double ref = referenceSumContiguous(values);
        assertEquals(ref, fast.getData()[0], 1e-6);
        assertEquals(ref, kahan.getData()[0], 1e-9);
        assertEquals(ref, neumaier.getData()[0], 1e-9);
    }

    private static double referenceSumContiguous(double[] values) {
        double acc = 0.0;
        for (double v : values) {
            acc += v;
        }
        return acc;
    }
}
