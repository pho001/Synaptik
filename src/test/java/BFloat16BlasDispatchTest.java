import backend.blas.BlasProvider;
import backend.blas.OpenBlasFfmBridge;
import backend.runtime.ExecutionMode;
import config.backend.KernelTuningConfig;
import config.optimizer.Conv2dLoweringConfig;
import config.optimizer.Conv2dLoweringMode;
import config.optimizer.OptimizerConfig;
import config.optimizer.OptimizerStage;
import config.optimizer.RewriteConfig;
import config.runtime.BlasConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import graph.execution.trace.ExecutionStepTrace;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class BFloat16BlasDispatchTest {
    @Test
    void bfloat16MatmulUsesBlasWhenEnabled() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isAvailable(), "OpenBLAS FFM is unavailable");

        Tensor a = new Tensor(random(64 * 64), new int[]{64, 64}, null, "a", DataType.BFLOAT16);
        Tensor b = new Tensor(random(64 * 64), new int[]{64, 64}, null, "b", DataType.BFLOAT16);
        Tensor out = a.matmul(b);

        var trace = CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults())
                .executeTraced(blasRuntime(1L), ExecutionMode.FORWARD);

        ExecutionStepTrace matmul = trace.steps().stream()
                .filter(step -> "MATMUL".equals(step.opType()))
                .findFirst()
                .orElse(null);
        assertNotNull(matmul);
        assertNotNull(matmul.metadata().matMul());
        assertTrue(matmul.metadata().matMul().useBlas());
    }

    @Test
    void bfloat16LinearUsesBlasWhenEnabled() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isAvailable(), "OpenBLAS FFM is unavailable");

        Tensor x = new Tensor(random(32 * 64), new int[]{32, 64}, null, "x", DataType.BFLOAT16);
        Tensor w = new Tensor(random(64 * 96), new int[]{64, 96}, null, "w", DataType.BFLOAT16);
        Tensor b = new Tensor(random(96), new int[]{96}, null, "b", DataType.BFLOAT16);
        Tensor out = x.linear(w, b).sum();

        var trace = CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults())
                .executeTraced(blasRuntime(1L), ExecutionMode.FORWARD);

        ExecutionStepTrace linear = trace.steps().stream()
                .filter(step -> "LINEAR".equals(step.opType()))
                .findFirst()
                .orElse(null);
        assertNotNull(linear);
        assertNotNull(linear.metadata().matMul());
        assertTrue(linear.metadata().matMul().useBlas());
    }

    @Test
    void bfloat16Conv2dLoweringBuildsConv2dGemmStep() {
        Tensor input = new Tensor(random(2 * 64 * 32 * 32), new int[]{2, 64, 32, 32}, null, "input", DataType.BFLOAT16);
        Tensor weight = new Tensor(random(128 * 64 * 3 * 3), new int[]{128, 64, 3, 3}, null, "weight", DataType.BFLOAT16);
        Tensor out = input.conv2d(weight, tensor.options.Conv2dOptions.defaults().withPadding(1, 1)).sum();

        OptimizerConfig optimizer = new OptimizerConfig(
                List.of(OptimizerStage.AR),
                new RewriteConfig(new Conv2dLoweringConfig(Conv2dLoweringMode.ALWAYS)),
                config.optimizer.CseConfig.strictDefaults(),
                config.optimizer.FuseConfig.inferenceDefaults(),
                config.optimizer.MemoryConfig.defaults()
        );

        var trace = CompiledGraph.compile(out, optimizer)
                .executeTraced(blasRuntime(1L), ExecutionMode.FORWARD);

        assertTrue(trace.steps().stream().anyMatch(step -> "CONV2D_GEMM".equals(step.opType())));
    }

    private static RuntimeConfig blasRuntime(long minWork) {
        return new RuntimeConfig(
                KernelTuningConfig.defaultsInference(),
                config.runtime.ApproximationConfig.defaults(),
                new BlasConfig(
                        BlasProvider.OPENBLAS_FFM,
                        minWork,
                        false,
                        100.0d,
                        false,
                        1
                )
        );
    }

    private static double[] random(int size) {
        java.util.Random random = new java.util.Random(7);
        double[] out = new double[size];
        for (int i = 0; i < size; i++) {
            out[i] = Math.sin(i * 0.031) + (random.nextDouble() - 0.5) * 0.1;
        }
        return out;
    }
}
