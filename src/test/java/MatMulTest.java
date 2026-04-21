import backend.runtime.ExecutionMode;
import backend.blas.BlasProvider;
import backend.blas.OpenBlasFfmBridge;
import config.backend.KernelTuningConfig;
import config.optimizer.OptimizerConfig;
import config.runtime.RuntimeConfig;
import config.runtime.BlasConfig;
import config.backend.CpuKernelConfig;
import graph.CompiledGraph;
import tensor.DataType;
import tensor.Tensor;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class MatMulTest {
    @Test
    void matMulForwardAndBackwardFloat64() {
        Tensor a = new Tensor(new double[]{1, 2, 3, 4}, new int[]{2, 2}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{5, 6, 7, 8}, new int[]{2, 2}, null, "b", DataType.FLOAT64);
        a.setRequiresGrad(true);
        b.setRequiresGrad(true);

        Tensor c = a.matmul(b);
        CompiledGraph graph = CompiledGraph.compile(c, OptimizerConfig.noOptimization());
        graph.execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD);
        assertArrayEquals(new double[]{19, 22, 43, 50}, c.toDoubleArrayCopy(), 1e-9);

        graph.execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);
        assertArrayEquals(new double[]{11, 15, 11, 15}, a.getGradient().toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{4, 4, 6, 6}, b.getGradient().toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void matMulForwardFloat32TiledParallel() {
        RuntimeConfig runtimeConfig = new RuntimeConfig(new CpuKernelConfig(
                4, 1, 2, 1,
                1, 1,
                1_000_000_000
        ), config.runtime.ApproximationConfig.defaults(), config.runtime.BlasConfig.disabled());

        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{7f, 8f, 9f, 10f, 11f, 12f}, new int[]{3, 2}, null, "b", DataType.FLOAT32);
        Tensor c = a.matmul(b);
        CompiledGraph.compile(c, OptimizerConfig.noOptimization()).execute(runtimeConfig, ExecutionMode.FORWARD);

        assertArrayEquals(new double[]{58, 64, 139, 154}, c.toDoubleArrayCopy(), 1e-5);
    }

    @Test
    void matMulShapeMismatchThrows() {
        Tensor a = new Tensor(new double[]{1, 2, 3, 4}, new int[]{2, 2}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{3, 2}, null, "b", DataType.FLOAT64);
        assertThrows(IllegalArgumentException.class, () -> a.matmul(b));
    }

    @Test
    void batchedMatMulForwardFloat64() {
        Tensor a = new Tensor(new double[]{
                1, 2, 3, 4,
                5, 6, 7, 8
        }, new int[]{2, 2, 2}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{
                1, 2,
                3, 4
        }, new int[]{2, 2, 1}, null, "b", DataType.FLOAT64);

        Tensor out = a.matmul(b);
        CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new int[]{2, 2, 1}, out.getShape());
        assertArrayEquals(new double[]{5, 11, 39, 53}, out.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void batchedMatMulBroadcastBatchDimensions() {
        Tensor a = new Tensor(new double[]{1, 2, 3, 4}, new int[]{1, 2, 2}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{
                1, 2,
                3, 4
        }, new int[]{2, 2, 1}, null, "b", DataType.FLOAT64);

        Tensor out = a.matmul(b);
        CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(new int[]{2, 2, 1}, out.getShape());
        assertArrayEquals(new double[]{5, 11, 11, 25}, out.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void bfloat16MatmulThenReluMatchesBaselineWhenBlasContinuationIsEnabled() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isAvailable(), "OpenBLAS FFM is unavailable");

        double[] aValues = random(64 * 64, 71);
        double[] bValues = random(64 * 96, 73);

        Tensor aBase = new Tensor(aValues.clone(), new int[]{64, 64}, null, "aBase", DataType.FLOAT64);
        Tensor bBase = new Tensor(bValues.clone(), new int[]{64, 96}, null, "bBase", DataType.FLOAT64);
        Tensor baseline = aBase.matmul(bBase).relu();
        CompiledGraph.compile(baseline, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
        double[] expected = baseline.toDoubleArrayCopy().clone();

        Tensor a = new Tensor(aValues.clone(), new int[]{64, 64}, null, "a", DataType.BFLOAT16);
        Tensor b = new Tensor(bValues.clone(), new int[]{64, 96}, null, "b", DataType.BFLOAT16);
        Tensor out = a.matmul(b).relu();
        CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults())
                .execute(bfloat16BlasRuntime(), ExecutionMode.FORWARD);

        assertArrayEquals(expected, out.toDoubleArrayCopy(), 2e-2);
    }

    @Test
    void bfloat16MatmulThenAddMatchesBaselineWhenBlasContinuationIsEnabled() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isAvailable(), "OpenBLAS FFM is unavailable");

        double[] aValues = random(64 * 64, 81);
        double[] bValues = random(64 * 96, 83);
        double[] cValues = random(64 * 96, 89);

        Tensor aBase = new Tensor(aValues.clone(), new int[]{64, 64}, null, "aBase", DataType.FLOAT64);
        Tensor bBase = new Tensor(bValues.clone(), new int[]{64, 96}, null, "bBase", DataType.FLOAT64);
        Tensor cBase = new Tensor(cValues.clone(), new int[]{64, 96}, null, "cBase", DataType.FLOAT64);
        Tensor baseline = aBase.matmul(bBase).add(cBase);
        CompiledGraph.compile(baseline, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
        double[] expected = baseline.toDoubleArrayCopy().clone();

        Tensor a = new Tensor(aValues.clone(), new int[]{64, 64}, null, "a", DataType.BFLOAT16);
        Tensor b = new Tensor(bValues.clone(), new int[]{64, 96}, null, "b", DataType.BFLOAT16);
        Tensor c = new Tensor(cValues.clone(), new int[]{64, 96}, null, "c", DataType.BFLOAT16);
        Tensor out = a.matmul(b).add(c);
        CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults())
                .execute(bfloat16BlasRuntime(), ExecutionMode.FORWARD);

        assertArrayEquals(expected, out.toDoubleArrayCopy(), 2e-2);
    }

    @Test
    void bfloat16MatmulThenFusedNumericChainMatchesBaselineWhenBlasContinuationIsEnabled() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isAvailable(), "OpenBLAS FFM is unavailable");

        double[] aValues = random(64 * 64, 91);
        double[] bValues = random(64 * 96, 97);

        Tensor aBase = new Tensor(aValues.clone(), new int[]{64, 64}, null, "aBase", DataType.FLOAT64);
        Tensor bBase = new Tensor(bValues.clone(), new int[]{64, 96}, null, "bBase", DataType.FLOAT64);
        Tensor baseline = aBase.matmul(bBase).relu().abs().clampMax(1.0);
        CompiledGraph.compile(baseline, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
        double[] expected = baseline.toDoubleArrayCopy().clone();

        Tensor a = new Tensor(aValues.clone(), new int[]{64, 64}, null, "a", DataType.BFLOAT16);
        Tensor b = new Tensor(bValues.clone(), new int[]{64, 96}, null, "b", DataType.BFLOAT16);
        Tensor out = a.matmul(b).relu().abs().clampMax(1.0);
        CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults())
                .execute(bfloat16BlasRuntime(), ExecutionMode.FORWARD);

        assertArrayEquals(expected, out.toDoubleArrayCopy(), 2e-2);
    }

    @Test
    void bfloat16MatmulReshapeThenNegMatchesBaselineWhenJavaContinuationIsEnabled() {
        double[] aValues = random(64 * 64, 111);
        double[] bValues = random(64 * 96, 113);

        Tensor aBase = new Tensor(aValues.clone(), new int[]{64, 64}, null, "aBase", DataType.BFLOAT16);
        Tensor bBase = new Tensor(bValues.clone(), new int[]{64, 96}, null, "bBase", DataType.BFLOAT16);
        Tensor baseline = aBase.matmul(bBase).reshape(32, 192).neg();
        CompiledGraph.compile(baseline, OptimizerConfig.noOptimization())
                .execute(bfloat16JavaRuntime(), ExecutionMode.FORWARD);
        double[] expected = baseline.toDoubleArrayCopy().clone();

        Tensor a = new Tensor(aValues.clone(), new int[]{64, 64}, null, "a", DataType.BFLOAT16);
        Tensor b = new Tensor(bValues.clone(), new int[]{64, 96}, null, "b", DataType.BFLOAT16);
        Tensor out = a.matmul(b).reshape(32, 192).neg();
        CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults())
                .execute(bfloat16JavaRuntime(), ExecutionMode.FORWARD);

        assertArrayEquals(expected, out.toDoubleArrayCopy(), 1e-6);
    }

    @Test
    void bfloat16MatmulJavaFallbackMatchesFloat64Baseline() {
        double[] aValues = random(32 * 48, 101);
        double[] bValues = random(48 * 24, 103);

        Tensor aBase = new Tensor(aValues.clone(), new int[]{32, 48}, null, "aBase", DataType.FLOAT64);
        Tensor bBase = new Tensor(bValues.clone(), new int[]{48, 24}, null, "bBase", DataType.FLOAT64);
        Tensor baseline = aBase.matmul(bBase);
        CompiledGraph.compile(baseline, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
        double[] expected = baseline.toDoubleArrayCopy().clone();

        Tensor a = new Tensor(aValues.clone(), new int[]{32, 48}, null, "a", DataType.BFLOAT16);
        Tensor b = new Tensor(bValues.clone(), new int[]{48, 24}, null, "b", DataType.BFLOAT16);
        Tensor out = a.matmul(b);
        CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .execute(bfloat16JavaRuntime(), ExecutionMode.FORWARD);

        assertArrayEquals(expected, out.toDoubleArrayCopy(), 3e-2);
    }

    @Test
    void bfloat16UnaryProducerCanFeedMatmulViaJavaContinuation() {
        double[] aValues = random(64 * 64, 121);
        double[] bValues = random(64 * 96, 123);

        Tensor aBase = new Tensor(aValues.clone(), new int[]{64, 64}, null, "aBase", DataType.BFLOAT16);
        Tensor bBase = new Tensor(bValues.clone(), new int[]{64, 96}, null, "bBase", DataType.BFLOAT16);
        Tensor baseline = aBase.relu().matmul(bBase);
        CompiledGraph.compile(baseline, OptimizerConfig.noOptimization())
                .execute(bfloat16JavaRuntime(), ExecutionMode.FORWARD);
        double[] expected = baseline.toDoubleArrayCopy().clone();

        Tensor a = new Tensor(aValues.clone(), new int[]{64, 64}, null, "a", DataType.BFLOAT16);
        Tensor b = new Tensor(bValues.clone(), new int[]{64, 96}, null, "b", DataType.BFLOAT16);
        Tensor out = a.relu().matmul(b);
        CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults())
                .execute(bfloat16JavaRuntime(), ExecutionMode.FORWARD);

        assertArrayEquals(expected, out.toDoubleArrayCopy(), 1e-6);
    }

    @Test
    void bfloat16DualUnaryProducersCanFeedMatmulViaJavaContinuation() {
        double[] aValues = random(64 * 64, 131);
        double[] bValues = random(64 * 96, 137);

        Tensor aBase = new Tensor(aValues.clone(), new int[]{64, 64}, null, "aBase", DataType.BFLOAT16);
        Tensor bBase = new Tensor(bValues.clone(), new int[]{64, 96}, null, "bBase", DataType.BFLOAT16);
        Tensor baseline = aBase.relu().matmul(bBase.relu());
        CompiledGraph.compile(baseline, OptimizerConfig.noOptimization())
                .execute(bfloat16JavaRuntime(), ExecutionMode.FORWARD);
        double[] expected = baseline.toDoubleArrayCopy().clone();

        Tensor a = new Tensor(aValues.clone(), new int[]{64, 64}, null, "a", DataType.BFLOAT16);
        Tensor b = new Tensor(bValues.clone(), new int[]{64, 96}, null, "b", DataType.BFLOAT16);
        Tensor out = a.relu().matmul(b.relu());
        CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults())
                .execute(bfloat16JavaRuntime(), ExecutionMode.FORWARD);

        assertArrayEquals(expected, out.toDoubleArrayCopy(), 1e-6);
    }

    private static RuntimeConfig bfloat16BlasRuntime() {
        return new RuntimeConfig(
                KernelTuningConfig.defaultsInference(),
                config.runtime.ApproximationConfig.defaults(),
                new BlasConfig(
                        BlasProvider.OPENBLAS_FFM,
                        1L,
                        false,
                        100.0d,
                        false,
                        1
                )
        );
    }

    private static RuntimeConfig bfloat16JavaRuntime() {
        return new RuntimeConfig(
                KernelTuningConfig.defaultsInference(),
                config.runtime.ApproximationConfig.defaults(),
                BlasConfig.disabled()
        );
    }

    private static double[] random(int size, int seed) {
        java.util.Random random = new java.util.Random(seed);
        double[] out = new double[size];
        for (int i = 0; i < size; i++) {
            out[i] = Math.sin(i * 0.031) + (random.nextDouble() - 0.5) * 0.1;
        }
        return out;
    }
}
