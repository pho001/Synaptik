import backend.runtime.ExecutionMode;
import backend.blas.BlasProvider;
import backend.blas.OpenBlasFfmBridge;
import config.backend.KernelTuningConfig;
import config.compile.CompileConfig;
import config.runtime.BlasConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import graph.execution.PreparedExecution;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class LinearExecutionTest {

    @Test
    void linearForwardWithoutBiasMatchesMatmul() {
        Tensor input = new Tensor(new double[]{
                1, 2,
                3, 4
        }, new int[]{2, 2}, null, "input", DataType.FLOAT64);
        Tensor weight = new Tensor(new double[]{
                5, 6,
                7, 8
        }, new int[]{2, 2}, null, "weight", DataType.FLOAT64);

        Tensor out = input.linear(weight);
        CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        assertArrayEquals(new int[]{2, 2}, out.getShape());
        assertArrayEquals(new double[]{19, 22, 43, 50}, out.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void linearForwardWithBiasBroadcastsAcrossLeadingDimensions() {
        Tensor input = new Tensor(new double[]{
                1, 2,
                3, 4,
                5, 6,
                7, 8
        }, new int[]{2, 2, 2}, null, "input", DataType.FLOAT64);
        Tensor weight = new Tensor(new double[]{
                1, 10,
                100, 1000
        }, new int[]{2, 2}, null, "weight", DataType.FLOAT64);
        Tensor bias = new Tensor(new double[]{0.5, -0.5}, new int[]{2}, null, "bias", DataType.FLOAT64);

        Tensor out = input.linear(weight, bias);
        CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        assertArrayEquals(new int[]{2, 2, 2}, out.getShape());
        assertArrayEquals(new double[]{
                201.5, 2009.5,
                403.5, 4029.5,
                605.5, 6049.5,
                807.5, 8069.5
        }, out.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void preparedLinearInvalidatesPackedWeightCacheWhenWeightChanges() {
        Tensor input = new Tensor(new float[]{
                1, 2,
                3, 4
        }, new int[]{2, 2}, null, "input", DataType.FLOAT32);
        Tensor weight = new Tensor(new float[]{
                5, 6,
                7, 8
        }, new int[]{2, 2}, null, "weight", DataType.FLOAT32);

        Tensor out = input.linear(weight);
        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults());

        execution.execute(ExecutionMode.FORWARD);
        assertArrayEquals(new double[]{19, 22, 43, 50}, out.toDoubleArrayCopy(), 1e-6);

        weight.setFloat32Data(new float[]{
                1, 0,
                0, 1
        });

        execution.execute(ExecutionMode.FORWARD);
        assertArrayEquals(new double[]{1, 2, 3, 4}, out.toDoubleArrayCopy(), 1e-6);
    }

    @Test
    void linearBackwardUsesMatmulAndBiasReductionContracts() {
        Tensor input = new Tensor(new double[]{
                1, 2,
                3, 4
        }, new int[]{2, 2}, null, "input", DataType.FLOAT64);
        Tensor weight = new Tensor(new double[]{
                5, 6,
                7, 8
        }, new int[]{2, 2}, null, "weight", DataType.FLOAT64);
        Tensor bias = new Tensor(new double[]{1, 2}, new int[]{2}, null, "bias", DataType.FLOAT64);
        input.setRequiresGrad(true);
        weight.setRequiresGrad(true);
        bias.setRequiresGrad(true);

        Tensor loss = input.linear(weight, bias).sum();
        CompiledGraph.compile(loss, CompileConfig.training())
                .prepare(RuntimeConfig.trainingDefaults()).execute(ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(new double[]{11, 15, 11, 15}, input.getGradient().toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{4, 4, 6, 6}, weight.getGradient().toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{2, 2}, bias.getGradient().toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void linearRejectsInvalidWeightAndBiasShapes() {
        Tensor input = new Tensor(new double[]{1, 2, 3, 4}, new int[]{2, 2}, null, "input", DataType.FLOAT64);
        Tensor badWeight = new Tensor(new double[]{1, 2, 3, 4, 5, 6}, new int[]{3, 2}, null, "weight", DataType.FLOAT64);
        Tensor goodWeight = new Tensor(new double[]{1, 2, 3, 4}, new int[]{2, 2}, null, "weight", DataType.FLOAT64);
        Tensor badBias = new Tensor(new double[]{1, 2, 3}, new int[]{3}, null, "bias", DataType.FLOAT64);

        assertThrows(IllegalArgumentException.class, () -> input.linear(badWeight));
        assertThrows(IllegalArgumentException.class, () -> input.linear(goodWeight, badBias));
    }

    @Test
    void bfloat16LinearWithBiasMatchesBaselineWhenBlasIsEnabled() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isBFloat16OutputGemmAvailable(), "OpenBLAS BGEMM is unavailable");

        double[] inputValues = random(32 * 64, 11);
        double[] weightValues = random(64 * 96, 17);
        double[] biasValues = random(96, 23);

        Tensor inputBase = new Tensor(inputValues.clone(), new int[]{32, 64}, null, "inputBase", DataType.BFLOAT16);
        Tensor weightBase = new Tensor(weightValues.clone(), new int[]{64, 96}, null, "weightBase", DataType.BFLOAT16);
        Tensor biasBase = new Tensor(biasValues.clone(), new int[]{96}, null, "biasBase", DataType.BFLOAT16);
        Tensor baseline = inputBase.linear(weightBase, biasBase);
        CompiledGraph.compile(baseline, CompileConfig.noGraphOptimizationBaseline())
                .prepare(bfloat16JavaRuntime()).execute(ExecutionMode.FORWARD);
        double[] expected = baseline.toDoubleArrayCopy().clone();

        Tensor input = new Tensor(inputValues.clone(), new int[]{32, 64}, null, "input", DataType.BFLOAT16);
        Tensor weight = new Tensor(weightValues.clone(), new int[]{64, 96}, null, "weight", DataType.BFLOAT16);
        Tensor bias = new Tensor(biasValues.clone(), new int[]{96}, null, "bias", DataType.BFLOAT16);
        Tensor out = input.linear(weight, bias);
        CompiledGraph.compile(out, CompileConfig.inference())
                .prepare(bfloat16BlasRuntime()).execute(ExecutionMode.FORWARD);

        // BF16 BLAS and Java fallback both use BF16 inputs with F32 accumulation, but the BLAS
        // kernel may accumulate in a different order before the final BF16 publication.
        assertArrayEquals(expected, out.toDoubleArrayCopy(), 3e-2);
    }

    @Test
    void bfloat16LinearThenReluMatchesBaselineWhenBlasContinuationIsEnabled() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isBFloat16ToFloatGemmAvailable(), "OpenBLAS SBGEMM is unavailable");

        double[] inputValues = random(32 * 64, 31);
        double[] weightValues = random(64 * 96, 37);
        double[] biasValues = random(96, 41);

        Tensor inputBase = new Tensor(inputValues.clone(), new int[]{32, 64}, null, "inputBase", DataType.BFLOAT16);
        Tensor weightBase = new Tensor(weightValues.clone(), new int[]{64, 96}, null, "weightBase", DataType.BFLOAT16);
        Tensor biasBase = new Tensor(biasValues.clone(), new int[]{96}, null, "biasBase", DataType.BFLOAT16);
        Tensor baseline = inputBase.linear(weightBase, biasBase).relu();
        CompiledGraph.compile(baseline, CompileConfig.noGraphOptimizationBaseline())
                .prepare(bfloat16JavaRuntime()).execute(ExecutionMode.FORWARD);
        double[] expected = baseline.toDoubleArrayCopy().clone();

        Tensor input = new Tensor(inputValues.clone(), new int[]{32, 64}, null, "input", DataType.BFLOAT16);
        Tensor weight = new Tensor(weightValues.clone(), new int[]{64, 96}, null, "weight", DataType.BFLOAT16);
        Tensor bias = new Tensor(biasValues.clone(), new int[]{96}, null, "bias", DataType.BFLOAT16);
        Tensor out = input.linear(weight, bias).relu();
        CompiledGraph.compile(out, CompileConfig.inference())
                .prepare(bfloat16BlasRuntime()).execute(ExecutionMode.FORWARD);

        assertArrayEquals(expected, out.toDoubleArrayCopy(), 1e-6);
    }

    @Test
    void bfloat16LinearThenReluMatchesBaselineWhenJavaContinuationIsEnabled() {
        double[] inputValues = random(32 * 64, 51);
        double[] weightValues = random(64 * 96, 57);
        double[] biasValues = random(96, 61);

        Tensor inputBase = new Tensor(inputValues.clone(), new int[]{32, 64}, null, "inputBase", DataType.FLOAT64);
        Tensor weightBase = new Tensor(weightValues.clone(), new int[]{64, 96}, null, "weightBase", DataType.FLOAT64);
        Tensor biasBase = new Tensor(biasValues.clone(), new int[]{96}, null, "biasBase", DataType.FLOAT64);
        Tensor baseline = inputBase.linear(weightBase, biasBase).relu();
        CompiledGraph.compile(baseline, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);
        double[] expected = baseline.toDoubleArrayCopy().clone();

        Tensor input = new Tensor(inputValues.clone(), new int[]{32, 64}, null, "input", DataType.BFLOAT16);
        Tensor weight = new Tensor(weightValues.clone(), new int[]{64, 96}, null, "weight", DataType.BFLOAT16);
        Tensor bias = new Tensor(biasValues.clone(), new int[]{96}, null, "bias", DataType.BFLOAT16);
        Tensor out = input.linear(weight, bias).relu();
        CompiledGraph.compile(out, CompileConfig.inference())
                .prepare(bfloat16JavaRuntime()).execute(ExecutionMode.FORWARD);

        assertArrayEquals(expected, out.toDoubleArrayCopy(), 3e-2);
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
