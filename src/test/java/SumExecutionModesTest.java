import backend.runtime.ExecutionMode;
import config.optimizer.OptimizerConfig;
import config.runtime.RuntimeConfig;
import config.backend.CpuKernelConfig;
import config.backend.SumAccuracyMode;
import graph.CompiledGraph;
import tensor.Tensor;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

public class SumExecutionModesTest {
    private static final double EPS = 1e-9;

    @Test
    public void testSumAllParallelVectorMatchesReference() {
        RuntimeConfig runtimeConfig = runtimeConfig(new CpuKernelConfig(
                4, 32, 32, 32,
                1, 1, 0, 8, 256, 1_000_000_000, SumAccuracyMode.FAST
        ));

        double[] values = new double[20_000];
        for (int i = 0; i < values.length; i++) {
            values[i] = (i % 17) * 0.25 - 1.75;
        }

        Tensor a = new Tensor(values, new int[]{values.length}, null, "a");
        Tensor s = a.sum();
        CompiledGraph.compile(s, OptimizerConfig.noOptimization()).execute(runtimeConfig, ExecutionMode.FORWARD);

        assertEquals(referenceSumContiguous(values), s.toDoubleArrayCopy()[0], EPS);
    }

    @Test
    public void testSumAxisLastDimParallelVectorMatchesReference() {
        RuntimeConfig runtimeConfig = runtimeConfig(new CpuKernelConfig(
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
        CompiledGraph.compile(s, OptimizerConfig.noOptimization()).execute(runtimeConfig, ExecutionMode.FORWARD);

        double[] expected = new double[rows];
        for (int r = 0; r < rows; r++) {
            double acc = 0.0;
            int base = r * cols;
            for (int c = 0; c < cols; c++) {
                acc += values[base + c];
            }
            expected[r] = acc;
        }
        assertArrayEquals(expected, s.toDoubleArrayCopy(), EPS);
    }

    @Test
    public void testSumAxisKeepDimsPreservesReducedAxisAsSingleton() {
        Tensor a = new Tensor(new double[]{
                1, 2, 3,
                4, 5, 6
        }, new int[]{2, 3}, null, "matrix");

        Tensor s = a.sum(1, true);
        CompiledGraph.compile(s, OptimizerConfig.noOptimization()).execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new int[]{2, 1}, s.getShape());
        assertArrayEquals(new double[]{6.0, 15.0}, s.toDoubleArrayCopy(), EPS);
    }

    @Test
    public void testSumNonContiguousStridedVsMaterializedEquivalence() {
        Tensor a = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{2, 3}, new int[]{1, 2}, null, "a_noncontig");

        RuntimeConfig stridedConfig = runtimeConfig(new CpuKernelConfig(
                4, 32, 32, 32,
                1_024, 100_000, 0, 4, 4_096, 1_000_000_000, SumAccuracyMode.FAST
        ));
        Tensor stridedAll = a.sum();
        CompiledGraph.compile(stridedAll, OptimizerConfig.noOptimization()).execute(stridedConfig, ExecutionMode.FORWARD);
        Tensor stridedAxis = a.sum(1);
        CompiledGraph.compile(stridedAxis, OptimizerConfig.noOptimization()).execute(stridedConfig, ExecutionMode.FORWARD);

        RuntimeConfig materializedConfig = runtimeConfig(new CpuKernelConfig(
                4, 32, 32, 32,
                1_024, 100_000, 0, 4, 4_096, 0, SumAccuracyMode.FAST
        ));
        Tensor materializedAll = a.sum();
        CompiledGraph.compile(materializedAll, OptimizerConfig.noOptimization()).execute(materializedConfig, ExecutionMode.FORWARD);
        Tensor materializedAxis = a.sum(1);
        CompiledGraph.compile(materializedAxis, OptimizerConfig.noOptimization()).execute(materializedConfig, ExecutionMode.FORWARD);

        assertArrayEquals(stridedAll.toDoubleArrayCopy(), materializedAll.toDoubleArrayCopy(), EPS);
        assertArrayEquals(stridedAxis.toDoubleArrayCopy(), materializedAxis.toDoubleArrayCopy(), EPS);
        assertArrayEquals(new double[]{9.0, 12.0}, stridedAxis.toDoubleArrayCopy(), EPS);
    }

    @Test
    public void testSumAllAccuracyModesStayClose() {
        double[] values = new double[50_000];
        for (int i = 0; i < values.length; i++) {
            values[i] = (i % 2 == 0 ? 1.0 : -1.0) * (1e-6 * (i % 23));
        }

        Tensor fastTensor = new Tensor(values.clone(), new int[]{values.length}, null, "fast");
        RuntimeConfig fastConfig = runtimeConfig(new CpuKernelConfig(
                4, 32, 32, 32,
                1, 1, 0, 8, 512, 1_000_000_000, SumAccuracyMode.FAST
        ));
        Tensor fast = fastTensor.sum();
        CompiledGraph.compile(fast, OptimizerConfig.noOptimization()).execute(fastConfig, ExecutionMode.FORWARD);

        Tensor kahanTensor = new Tensor(values.clone(), new int[]{values.length}, null, "kahan");
        RuntimeConfig kahanConfig = runtimeConfig(new CpuKernelConfig(
                4, 32, 32, 32,
                1, 1, 0, 8, 512, 1_000_000_000, SumAccuracyMode.KAHAN
        ));
        Tensor kahan = kahanTensor.sum();
        CompiledGraph.compile(kahan, OptimizerConfig.noOptimization()).execute(kahanConfig, ExecutionMode.FORWARD);

        Tensor neumaierTensor = new Tensor(values.clone(), new int[]{values.length}, null, "neumaier");
        RuntimeConfig neumaierConfig = runtimeConfig(new CpuKernelConfig(
                4, 32, 32, 32,
                1, 1, 0, 8, 512, 1_000_000_000, SumAccuracyMode.NEUMAIER
        ));
        Tensor neumaier = neumaierTensor.sum();
        CompiledGraph.compile(neumaier, OptimizerConfig.noOptimization()).execute(neumaierConfig, ExecutionMode.FORWARD);

        double ref = referenceSumContiguous(values);
        assertEquals(ref, fast.toDoubleArrayCopy()[0], 1e-6);
        assertEquals(ref, kahan.toDoubleArrayCopy()[0], 1e-9);
        assertEquals(ref, neumaier.toDoubleArrayCopy()[0], 1e-9);
    }

    private static double referenceSumContiguous(double[] values) {
        double acc = 0.0;
        for (double v : values) {
            acc += v;
        }
        return acc;
    }

    private static RuntimeConfig runtimeConfig(CpuKernelConfig cpuKernelConfig) {
        return new RuntimeConfig(cpuKernelConfig, config.runtime.ApproximationConfig.defaults(), config.runtime.BlasConfig.disabled());
    }
}
