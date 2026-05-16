import backend.blas.BlasProvider;
import backend.blas.OpenBlasFfmBridge;
import backend.runtime.ExecutionMode;
import config.backend.KernelTuningConfig;
import config.compile.BackendPlanningConfig;
import config.compile.CompileConfig;
import config.compile.GraphOptimizationConfig;
import config.optimizer.Conv2dLoweringConfig;
import config.optimizer.Conv2dLoweringMode;
import config.optimizer.RewriteConfig;
import config.runtime.AcceleratorConfig;
import config.runtime.BlasConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import graph.execution.trace.ExecutionStepTrace;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class BFloat16BlasDispatchTest {
    @Test
    void bfloat16MatmulUsesBlasWhenEnabled() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isBFloat16OutputGemmAvailable(), "OpenBLAS BGEMM is unavailable");

        Tensor a = new Tensor(random(64 * 64), new int[]{64, 64}, null, "a", DataType.BFLOAT16);
        Tensor b = new Tensor(random(64 * 64), new int[]{64, 64}, null, "b", DataType.BFLOAT16);
        Tensor out = a.matmul(b);

        var trace = CompiledGraph.compile(out, CompileConfig.inference())
                .executeTraced(blasRuntime(1L), ExecutionMode.FORWARD);

        ExecutionStepTrace matmul = trace.steps().stream()
                .filter(step -> "MATMUL".equals(step.opType()))
                .findFirst()
                .orElse(null);
        assertNotNull(matmul);
        assertNotNull(matmul.metadata().matMul());
        assertTrue(matmul.metadata().matMul().useBlas());
        assertEquals("cblas_bgemm", matmul.metadata().matMul().blasSymbol());
        assertEquals("BGEMM", matmul.metadata().matMul().bf16OutputRoute());
        assertEquals("", matmul.metadata().matMul().bf16ContinuationRoute());
        assertEquals("BF16_OUTPUT", matmul.metadata().matMul().bf16ComputePrecision());
        assertEquals("BF16", matmul.metadata().matMul().bf16OutputPrecision());
        assertEquals(true, matmul.metadata().matMul().openblasBgemmAvailable());
    }

    @Test
    void bfloat16LinearUsesBlasWhenEnabled() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isBFloat16ToFloatGemmAvailable(), "OpenBLAS SBGEMM is unavailable");

        Tensor x = new Tensor(random(32 * 64), new int[]{32, 64}, null, "x", DataType.BFLOAT16);
        Tensor w = new Tensor(random(64 * 96), new int[]{64, 96}, null, "w", DataType.BFLOAT16);
        Tensor b = new Tensor(random(96), new int[]{96}, null, "b", DataType.BFLOAT16);
        Tensor out = x.linear(w, b).relu().sum();

        var trace = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                .executeTraced(blasRuntime(1L), ExecutionMode.FORWARD);

        ExecutionStepTrace linear = trace.steps().stream()
                .filter(step -> "LINEAR".equals(step.opType()))
                .findFirst()
                .orElse(null);
        assertNotNull(linear);
        assertNotNull(linear.metadata().matMul());
        assertTrue(linear.metadata().matMul().useBlas());
        assertEquals("cblas_sbgemm", linear.metadata().matMul().blasSymbol());
        assertEquals("SBGEMM", linear.metadata().matMul().bf16ContinuationRoute());
        assertEquals("PROMOTED_F32", linear.metadata().matMul().bf16OutputRoute());
        assertEquals("F32_PROMOTED", linear.metadata().matMul().bf16ComputePrecision());
        assertEquals("F32", linear.metadata().matMul().bf16OutputPrecision());
        assertEquals(true, linear.metadata().matMul().openblasSbgemmAvailable());
    }

    @Test
    void bfloat16MatmulTraceReportsJavaRouteWhenBlasIsDisabled() {
        Tensor a = new Tensor(random(16 * 16), new int[]{16, 16}, null, "a", DataType.BFLOAT16);
        Tensor b = new Tensor(random(16 * 16), new int[]{16, 16}, null, "b", DataType.BFLOAT16);
        Tensor out = a.matmul(b);

        RuntimeConfig runtime = new RuntimeConfig(
                KernelTuningConfig.defaultsInference(),
                config.runtime.ApproximationConfig.defaults(),
                BlasConfig.disabled()
        );

        var trace = CompiledGraph.compile(out, CompileConfig.inference())
                .executeTraced(runtime, ExecutionMode.FORWARD);

        ExecutionStepTrace matmul = trace.steps().stream()
                .filter(step -> "MATMUL".equals(step.opType()))
                .findFirst()
                .orElse(null);
        assertNotNull(matmul);
        assertNotNull(matmul.metadata().matMul());
        assertEquals("JAVA_DIRECT", matmul.metadata().matMul().route());
        assertEquals("JAVA", matmul.metadata().matMul().bf16ContinuationRoute());
        assertEquals("JAVA", matmul.metadata().matMul().bf16OutputRoute());
        assertEquals("F32_PROMOTED", matmul.metadata().matMul().bf16ComputePrecision());
        assertEquals("BF16", matmul.metadata().matMul().bf16OutputPrecision());
        assertEquals(false, matmul.metadata().matMul().openblasSbgemmAvailable());
        assertEquals(false, matmul.metadata().matMul().openblasBgemmAvailable());
    }

    @Test
    void bfloat16WideMatmulUsesWideSpecificBlasHeuristic() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isBFloat16OutputGemmAvailable(), "OpenBLAS BGEMM is unavailable");

        Tensor a = new Tensor(random(256 * 256), new int[]{256, 256}, null, "a", DataType.BFLOAT16);
        Tensor b = new Tensor(random(256 * 2048), new int[]{256, 2048}, null, "b", DataType.BFLOAT16);
        Tensor out = a.matmul(b);

        var trace = CompiledGraph.compile(out, CompileConfig.inference())
                .executeTraced(blasRuntimeWide(1L, true, 4.0d, true, 12.0d), ExecutionMode.FORWARD);

        ExecutionStepTrace matmul = trace.steps().stream()
                .filter(step -> "MATMUL".equals(step.opType()))
                .findFirst()
                .orElse(null);
        assertNotNull(matmul);
        assertNotNull(matmul.metadata().matMul());
        assertTrue(matmul.metadata().matMul().useBlas());
    }

    @Test
    void bfloat16Conv2dLoweringBuildsConv2dGemmStep() {
        Tensor input = new Tensor(random(2 * 64 * 32 * 32), new int[]{2, 64, 32, 32}, null, "input", DataType.BFLOAT16);
        Tensor weight = new Tensor(random(128 * 64 * 3 * 3), new int[]{128, 64, 3, 3}, null, "weight", DataType.BFLOAT16);
        Tensor out = input.conv2d(weight, tensor.options.Conv2dOptions.defaults().withPadding(1, 1)).sum();

        CompileConfig optimizer = convLoweringOnlyConfig();

        var trace = CompiledGraph.compile(out, optimizer)
                .executeTraced(blasRuntime(1L), ExecutionMode.FORWARD);

        assertTrue(trace.steps().stream().anyMatch(step -> "CONV2D_GEMM".equals(step.opType())));
    }

    @Test
    void bfloat16Conv2dTraceReportsBlasUsageWhenEnabled() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isBFloat16ToFloatGemmAvailable(), "OpenBLAS SBGEMM is unavailable");

        Tensor input = new Tensor(random(2 * 64 * 32 * 32), new int[]{2, 64, 32, 32}, null, "input", DataType.BFLOAT16);
        Tensor weight = new Tensor(random(128 * 64 * 3 * 3), new int[]{128, 64, 3, 3}, null, "weight", DataType.BFLOAT16);
        Tensor out = input.conv2d(weight, tensor.options.Conv2dOptions.defaults().withPadding(1, 1)).sum();

        CompileConfig optimizer = convLoweringOnlyConfig();

        var trace = CompiledGraph.compile(out, optimizer)
                .executeTraced(blasRuntime(1L), ExecutionMode.FORWARD);

        ExecutionStepTrace conv = trace.steps().stream()
                .filter(step -> "CONV2D_GEMM".equals(step.opType()))
                .findFirst()
                .orElse(null);
        assertNotNull(conv);
        assertNotNull(conv.metadata().conv());
        assertTrue("GEMM".equals(conv.metadata().conv().executionKind()));
        assertTrue(conv.metadata().conv().blasUsed());
        assertTrue("OPENBLAS_FFM".equals(conv.metadata().conv().blasProvider()));
    }

    @Test
    void bfloat16Conv2dTraceReportsJavaFallbackWhenBlasDisabled() {
        Tensor input = new Tensor(random(2 * 64 * 32 * 32), new int[]{2, 64, 32, 32}, null, "input", DataType.BFLOAT16);
        Tensor weight = new Tensor(random(128 * 64 * 3 * 3), new int[]{128, 64, 3, 3}, null, "weight", DataType.BFLOAT16);
        Tensor out = input.conv2d(weight, tensor.options.Conv2dOptions.defaults().withPadding(1, 1)).sum();

        CompileConfig optimizer = convLoweringOnlyConfig();

        RuntimeConfig runtime = new RuntimeConfig(
                KernelTuningConfig.defaultsInference(),
                config.runtime.ApproximationConfig.defaults(),
                BlasConfig.disabled()
        );

        var trace = CompiledGraph.compile(out, optimizer)
                .executeTraced(runtime, ExecutionMode.FORWARD);

        ExecutionStepTrace conv = trace.steps().stream()
                .filter(step -> "CONV2D_GEMM".equals(step.opType()))
                .findFirst()
                .orElse(null);
        assertNotNull(conv);
        assertNotNull(conv.metadata().conv());
        assertTrue("GEMM".equals(conv.metadata().conv().executionKind()));
        assertTrue(!conv.metadata().conv().blasUsed());
        assertTrue("NONE".equals(conv.metadata().conv().blasProvider()));
    }

    @Test
    void blasDispatchRemainsAvailableWithAutoAcceleratorPlanning() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isBFloat16OutputGemmAvailable(), "OpenBLAS BGEMM is unavailable");

        Tensor a = new Tensor(random(64 * 64), new int[]{64, 64}, null, "a", DataType.BFLOAT16);
        Tensor b = new Tensor(random(64 * 96), new int[]{64, 96}, null, "b", DataType.BFLOAT16);
        Tensor out = a.matmul(b);

        var execution = CompiledGraph.compile(
                        out,
                        CompileConfig.inference().withBackendPlanning(BackendPlanningConfig.autoAccelerator())
                )
                .prepare(blasRuntime(1L).withAccelerator(AcceleratorConfig.disabled()));

        var matmul = execution.forwardSteps().stream()
                .filter(step -> step.node().getOperation() != null
                        && step.node().getOperation().opType() == operations.Operation.OpType.MATMUL)
                .findFirst()
                .orElseThrow();

        assertTrue("CPU_MATMUL_BLAS".equals(matmul.metadata().cpuPlan().computeContract().backend().name()));
    }

    private static CompileConfig convLoweringOnlyConfig() {
        return CompileConfig.inference()
                .withGraphOptimization(GraphOptimizationConfig
                        .stages(false, false, false, false, true)
                        .withRewrite(new RewriteConfig(new Conv2dLoweringConfig(Conv2dLoweringMode.ALWAYS))));
    }

    private static RuntimeConfig blasRuntime(long minWork) {
        return blasRuntimeWide(minWork, false, 100.0d, false, 100.0d);
    }

    private static RuntimeConfig blasRuntimeWide(
            long minWork,
            boolean requireMgeK,
            double maxNOverK,
            boolean wideRequireMgeK,
            double wideMaxNOverK
    ) {
        return new RuntimeConfig(
                KernelTuningConfig.defaultsInference(),
                config.runtime.ApproximationConfig.defaults(),
                new BlasConfig(
                        BlasProvider.OPENBLAS_FFM,
                        minWork,
                        requireMgeK,
                        maxNOverK,
                        wideRequireMgeK,
                        wideMaxNOverK,
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
