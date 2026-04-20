import backend.runtime.ExecutionMode;
import config.optimizer.OptimizerConfig;
import config.runtime.RuntimeConfig;
import config.backend.CpuKernelConfig;
import graph.CompiledGraph;
import tensor.DataType;
import tensor.Tensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

public class NonContiguousExecutionTest {
    private static final double EPS = 1e-9;
    private static final double EPS32 = 1e-6;

    @Test
    public void testAddNonContiguousStridedPath() {
        // size=6 < threshold(100): strided path should be selected
        RuntimeConfig runtimeConfig = runtimeConfig(new CpuKernelConfig(4, 32, 32, 32, 1_024, 100_000, 100));

        Tensor a = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, new int[]{1, 2}, null, "a_noncontig", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{10, 20, 30, 40, 50, 60}, new int[]{2, 3}, null, "b_contig", DataType.FLOAT64);

        Tensor c = a.add(b);
        CompiledGraph.compile(c, OptimizerConfig.noOptimization()).execute(runtimeConfig, ExecutionMode.FORWARD);

        double[] expected = add(remapToContiguous(a), b.toDoubleArrayCopy());
        assertArrayEquals(expected, c.toDoubleArrayCopy(), EPS);
    }

    @Test
    public void testAddNonContiguousMaterializePath() {
        // threshold(0): materialize path should be selected
        RuntimeConfig runtimeConfig = runtimeConfig(new CpuKernelConfig(4, 32, 32, 32, 1_024, 100_000, 0));

        Tensor a = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, new int[]{1, 2}, null, "a_noncontig", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{10, 20, 30, 40, 50, 60}, new int[]{2, 3}, null, "b_contig", DataType.FLOAT64);

        Tensor c = a.add(b);
        CompiledGraph.compile(c, OptimizerConfig.noOptimization()).execute(runtimeConfig, ExecutionMode.FORWARD);

        double[] expected = add(remapToContiguous(a), b.toDoubleArrayCopy());
        assertArrayEquals(expected, c.toDoubleArrayCopy(), EPS);
    }

    @Test
    public void testLogNonContiguousStridedVsMaterializeEquivalence() {
        Tensor a = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, new int[]{1, 2}, null, "a_noncontig", DataType.FLOAT64);

        RuntimeConfig stridedConfig = runtimeConfig(new CpuKernelConfig(4, 32, 32, 32, 1_024, 100_000, 100));
        Tensor s = a.log();
        CompiledGraph.compile(s, OptimizerConfig.noOptimization()).execute(stridedConfig, ExecutionMode.FORWARD);
        double[] strided = s.toDoubleArrayCopy().clone();

        RuntimeConfig materializedConfig = runtimeConfig(new CpuKernelConfig(4, 32, 32, 32, 1_024, 100_000, 0));
        Tensor m = a.log();
        CompiledGraph.compile(m, OptimizerConfig.noOptimization()).execute(materializedConfig, ExecutionMode.FORWARD);
        double[] materialized = m.toDoubleArrayCopy().clone();

        assertArrayEquals(strided, materialized, EPS);
    }

    @Test
    public void testAddBroadcastWithNonContiguousInput() {
        RuntimeConfig runtimeConfig = runtimeConfig(new CpuKernelConfig(4, 32, 32, 32, 1_024, 100_000, 100));

        Tensor a = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, new int[]{1, 2}, null, "a_noncontig", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{10, 20, 30}, new int[]{3}, null, "b_broadcast", DataType.FLOAT64);

        Tensor c = a.add(b);
        CompiledGraph.compile(c, OptimizerConfig.noOptimization()).execute(runtimeConfig, ExecutionMode.FORWARD);

        Tensor ref = a.contiguous().add(b);
        CompiledGraph.compile(ref, OptimizerConfig.noOptimization()).execute(runtimeConfig, ExecutionMode.FORWARD);
        assertArrayEquals(ref.toDoubleArrayCopy(), c.toDoubleArrayCopy(), EPS);
    }

    @Test
    public void testAddBroadcastRowViewFloat32StridedVsMaterializeEquivalence() {
        RuntimeConfig stridedConfig = runtimeConfig(new CpuKernelConfig(4, 32, 32, 32, 1_024, 100_000, 100));
        RuntimeConfig materializedConfig = runtimeConfig(new CpuKernelConfig(4, 32, 32, 32, 1_024, 100_000, 0));

        Tensor left = new Tensor(new float[]{1, 2, 3, 4, 5, 6, 7, 8}, new int[]{2, 4}, null, "left", DataType.FLOAT32);
        Tensor bias = new Tensor(new float[]{10, 20}, new int[]{2, 1}, null, "bias", DataType.FLOAT32);

        Tensor strided = left.add(bias.expand(2, 4));
        CompiledGraph.compile(strided, OptimizerConfig.noOptimization()).execute(stridedConfig, ExecutionMode.FORWARD);

        Tensor materialized = left.add(bias.expand(2, 4));
        CompiledGraph.compile(materialized, OptimizerConfig.noOptimization()).execute(materializedConfig, ExecutionMode.FORWARD);

        assertArrayEquals(materialized.toDoubleArrayCopy(), strided.toDoubleArrayCopy(), EPS32);
    }

    @Test
    public void testMulNonContiguousFloat32StridedVsMaterializeEquivalence() {
        RuntimeConfig stridedConfig = runtimeConfig(new CpuKernelConfig(4, 32, 32, 32, 1_024, 100_000, 100));
        RuntimeConfig materializedConfig = runtimeConfig(new CpuKernelConfig(4, 32, 32, 32, 1_024, 100_000, 0));

        Tensor left = new Tensor(new float[]{1, 2, 3, 4, 5, 6, 7, 8}, new int[]{2, 4}, new int[]{1, 2}, null, "left_noncontig", DataType.FLOAT32);
        Tensor right = new Tensor(new float[]{2, 3, 4, 5, 6, 7, 8, 9}, new int[]{2, 4}, null, "right_contig", DataType.FLOAT32);

        Tensor strided = left.mul(right);
        CompiledGraph.compile(strided, OptimizerConfig.noOptimization()).execute(stridedConfig, ExecutionMode.FORWARD);

        Tensor materialized = left.mul(right);
        CompiledGraph.compile(materialized, OptimizerConfig.noOptimization()).execute(materializedConfig, ExecutionMode.FORWARD);

        assertArrayEquals(materialized.toDoubleArrayCopy(), strided.toDoubleArrayCopy(), EPS32);
    }

    @Test
    public void testMulScalarExpandedRowViewFloat32StridedVsMaterializeEquivalence() {
        RuntimeConfig stridedConfig = runtimeConfig(new CpuKernelConfig(4, 32, 32, 32, 1_024, 100_000, 100));
        RuntimeConfig materializedConfig = runtimeConfig(new CpuKernelConfig(4, 32, 32, 32, 1_024, 100_000, 0));

        Tensor bias = new Tensor(new float[]{3, 7}, new int[]{2, 1}, null, "bias", DataType.FLOAT32);

        Tensor strided = bias.expand(2, 4).mul(0.5);
        CompiledGraph.compile(strided, OptimizerConfig.noOptimization()).execute(stridedConfig, ExecutionMode.FORWARD);

        Tensor materialized = bias.expand(2, 4).mul(0.5);
        CompiledGraph.compile(materialized, OptimizerConfig.noOptimization()).execute(materializedConfig, ExecutionMode.FORWARD);

        assertArrayEquals(materialized.toDoubleArrayCopy(), strided.toDoubleArrayCopy(), EPS32);
    }

    private static RuntimeConfig runtimeConfig(CpuKernelConfig cpuKernelConfig) {
        return new RuntimeConfig(cpuKernelConfig, config.runtime.ApproximationConfig.defaults(), config.runtime.BlasConfig.disabled());
    }

    private static double[] add(double[] left, double[] right) {
        double[] out = new double[left.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = left[i] + right[i];
        }
        return out;
    }

    private static double[] remapToContiguous(Tensor src) {
        return src.toDoubleArrayCopy();
    }
}
