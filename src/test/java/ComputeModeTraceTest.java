import backend.blas.OpenBlasFfmBridge;
import backend.runtime.ExecutionMode;
import config.compile.CompileConfig;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

public class ComputeModeTraceTest {
    @Test
    void bfloat16ElementWiseTraceCarriesBfloat16F32ComputeMode() {
        Tensor a = new Tensor(new double[]{1, 2, 3, 4}, new int[]{4}, null, "a", DataType.BFLOAT16);
        Tensor b = new Tensor(new double[]{5, 6, 7, 8}, new int[]{4}, null, "b", DataType.BFLOAT16);
        Tensor out = a.add(b);

        var trace = graph.CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                .executeTraced(config.runtime.RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        var add = trace.steps().stream()
                .filter(step -> "ADD".equals(step.opType()))
                .findFirst()
                .orElse(null);
        assertNotNull(add);
        assertNotNull(add.metadata().compute());
        assertEquals("F32", add.metadata().compute().mode());
        assertEquals("BFLOAT16", add.metadata().compute().storageType());
        assertEquals("CPU_ELEMENTWISE", add.metadata().compute().backend());
    }

    @Test
    void bfloat16ReductionTraceCarriesBfloat16F32ComputeMode() {
        Tensor x = new Tensor(new double[]{1, 2, 3, 4}, new int[]{4}, null, "x", DataType.BFLOAT16);
        Tensor out = x.sum();

        var trace = graph.CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                .executeTraced(config.runtime.RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        var sum = trace.steps().stream()
                .filter(step -> "SUM".equals(step.opType()))
                .findFirst()
                .orElse(null);
        assertNotNull(sum);
        assertNotNull(sum.metadata().compute());
        assertEquals("F32", sum.metadata().compute().mode());
        assertEquals("BFLOAT16", sum.metadata().compute().storageType());
        assertEquals("CPU_REDUCTION", sum.metadata().compute().backend());
    }

    @Test
    void bfloat16MatmulTraceCarriesBfloat16BlasComputeModeWhenBlasIsUsed() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isAvailable(), "OpenBLAS FFM is unavailable");

        Tensor a = new Tensor(random(64 * 64), new int[]{64, 64}, null, "a", DataType.BFLOAT16);
        Tensor b = new Tensor(random(64 * 64), new int[]{64, 64}, null, "b", DataType.BFLOAT16);
        Tensor out = a.matmul(b);

        var trace = graph.CompiledGraph.compile(out, CompileConfig.inference())
                .executeTraced(bfloat16BlasRuntime(), ExecutionMode.FORWARD);

        var matmul = trace.steps().stream()
                .filter(step -> "MATMUL".equals(step.opType()))
                .findFirst()
                .orElse(null);
        assertNotNull(matmul);
        assertNotNull(matmul.metadata().compute());
        assertEquals("F32", matmul.metadata().compute().mode());
        assertEquals("BFLOAT16", matmul.metadata().compute().storageType());
        assertEquals("CPU_MATMUL_BLAS", matmul.metadata().compute().backend());
        assertNotNull(matmul.metadata().matMul());
        assertEquals(true, matmul.metadata().matMul().useBlas());
    }

    private static config.runtime.RuntimeConfig bfloat16BlasRuntime() {
        return new config.runtime.RuntimeConfig(
                config.backend.KernelTuningConfig.defaultsInference(),
                config.runtime.ApproximationConfig.defaults(),
                new config.runtime.BlasConfig(
                        backend.blas.BlasProvider.OPENBLAS_FFM,
                        1L,
                        false,
                        100.0d,
                        false,
                        1
                )
        );
    }

    private static double[] random(int size) {
        java.util.Random random = new java.util.Random(13);
        double[] out = new double[size];
        for (int i = 0; i < size; i++) {
            out[i] = Math.sin(i * 0.019) + (random.nextDouble() - 0.5) * 0.05;
        }
        return out;
    }
}
