import backend.ComputeEngine;
import config.backend.CpuKernelConfig;
import graph.optimizer.GraphOptimizer;
import tensor.DataType;
import tensor.Tensor;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

public class NonContiguousExecutionTest {
    private static final double EPS = 1e-9;

    @AfterEach
    public void resetCpuConfig() {
        ComputeEngine.setCpuKernelConfig(CpuKernelConfig.defaultsTraining());
    }

    @Test
    public void testAddNonContiguousStridedPath() {
        // size=6 < threshold(100): strided path should be selected
        ComputeEngine.setCpuKernelConfig(new CpuKernelConfig(4, 32, 32, 32, 1_024, 100_000, 0, 4, 4_096, 100));

        Tensor a = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, new int[]{1, 2}, null, "a_noncontig", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{10, 20, 30, 40, 50, 60}, new int[]{2, 3}, null, "b_contig", DataType.FLOAT64);

        Tensor c = a.add(b);
        c.compute(new GraphOptimizer());

        double[] expected = add(remapToContiguous(a), b.toDoubleArrayCopy());
        assertArrayEquals(expected, c.toDoubleArrayCopy(), EPS);
    }

    @Test
    public void testAddNonContiguousMaterializePath() {
        // threshold(0): materialize path should be selected
        ComputeEngine.setCpuKernelConfig(new CpuKernelConfig(4, 32, 32, 32, 1_024, 100_000, 0, 4, 4_096, 0));

        Tensor a = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, new int[]{1, 2}, null, "a_noncontig", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{10, 20, 30, 40, 50, 60}, new int[]{2, 3}, null, "b_contig", DataType.FLOAT64);

        Tensor c = a.add(b);
        c.compute(new GraphOptimizer());

        double[] expected = add(remapToContiguous(a), b.toDoubleArrayCopy());
        assertArrayEquals(expected, c.toDoubleArrayCopy(), EPS);
    }

    @Test
    public void testLogNonContiguousStridedVsMaterializeEquivalence() {
        Tensor a = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, new int[]{1, 2}, null, "a_noncontig", DataType.FLOAT64);

        ComputeEngine.setCpuKernelConfig(new CpuKernelConfig(4, 32, 32, 32, 1_024, 100_000, 0, 4, 4_096, 100));
        Tensor s = a.log();
        s.compute(new GraphOptimizer());
        double[] strided = s.toDoubleArrayCopy().clone();

        ComputeEngine.setCpuKernelConfig(new CpuKernelConfig(4, 32, 32, 32, 1_024, 100_000, 0, 4, 4_096, 0));
        Tensor m = a.log();
        m.compute(new GraphOptimizer());
        double[] materialized = m.toDoubleArrayCopy().clone();

        assertArrayEquals(strided, materialized, EPS);
    }

    @Test
    public void testAddBroadcastWithNonContiguousInput() {
        ComputeEngine.setCpuKernelConfig(new CpuKernelConfig(4, 32, 32, 32, 1_024, 100_000, 0, 4, 4_096, 100));

        Tensor a = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, new int[]{1, 2}, null, "a_noncontig", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{10, 20, 30}, new int[]{3}, null, "b_broadcast", DataType.FLOAT64);

        Tensor c = a.add(b);
        c.compute(new GraphOptimizer());

        Tensor ref = a.contiguous().add(b);
        ref.compute(new GraphOptimizer());
        assertArrayEquals(ref.toDoubleArrayCopy(), c.toDoubleArrayCopy(), EPS);
    }

    private static double[] add(double[] left, double[] right) {
        double[] out = new double[left.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = left[i] + right[i];
        }
        return out;
    }

    private static double[] remapToContiguous(Tensor src) {
        int[] shape = src.getShape();
        int[] srcStrides = src.getStrides();
        int[] dstStrides = contiguousStrides(shape);
        double[] srcData = src.toDoubleArrayCopy();
        double[] out = new double[srcData.length];

        for (int i = 0; i < out.length; i++) {
            int idx = i;
            int srcFlat = 0;
            for (int d = 0; d < shape.length; d++) {
                int coord = idx / dstStrides[d];
                idx %= dstStrides[d];
                srcFlat += coord * srcStrides[d];
            }
            out[i] = srcData[srcFlat];
        }
        return out;
    }

    private static int[] contiguousStrides(int[] shape) {
        int[] strides = new int[shape.length];
        int stride = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            strides[i] = stride;
            stride *= shape[i];
        }
        return strides;
    }
}
