package backend.cpu.nativecpu;

import backend.blas.BlasProvider;
import backend.blas.OpenBlasFfmBridge;
import backend.cpu.kernels.linalg.matmul.plan.MatMulExecutionRoute;
import backend.runtime.ExecutionMode;
import config.backend.KernelTuningConfig;
import config.compile.CompileConfig;
import config.runtime.ApproximationConfig;
import config.runtime.BlasConfig;
import config.runtime.BlasStorageMode;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import operations.Operation;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class NativeOpenBlasPlannerTest {
    @Test
    void cpuArrayStorageModeKeepsExistingArrayCopyBlasRoute() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isAvailable(), OpenBlasFfmBridge.unavailableReason());

        var step = matmulStep(DataType.FLOAT32, runtime(BlasStorageMode.CPU_ARRAY, 1L));

        assertEquals(MatMulExecutionRoute.OPENBLAS_ARRAY_COPYING, step.metadata().cpuPlan().matMulHints().route());
        assertEquals("F32BlasMatMulExecutable", step.metadata().cpuPlan().matMulExecutable().getClass().getSimpleName());
    }

    @Test
    void cpuNativeStorageModeSelectsFloat32NativeSegmentRouteEvenBelowArrayThreshold() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        var step = matmulStep(DataType.FLOAT32, runtime(BlasStorageMode.CPU_NATIVE, Long.MAX_VALUE));

        assertEquals(MatMulExecutionRoute.OPENBLAS_NATIVE_SEGMENT, step.metadata().cpuPlan().matMulHints().route());
        assertEquals("F32NativeBlasMatMulExecutable", step.metadata().cpuPlan().matMulExecutable().getClass().getSimpleName());
    }

    @Test
    void autoStorageModeSelectsFloat64NativeSegmentRouteForEligibleDenseMatmul() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat64GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        var step = matmulStep(DataType.FLOAT64, runtime(BlasStorageMode.AUTO, 1L));

        assertEquals(MatMulExecutionRoute.OPENBLAS_NATIVE_SEGMENT, step.metadata().cpuPlan().matMulHints().route());
        assertEquals("F64NativeBlasMatMulExecutable", step.metadata().cpuPlan().matMulExecutable().getClass().getSimpleName());
    }

    @Test
    void bfloat16SelectsNativeSegmentRouteWhenBgemmIsAvailable() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isBFloat16OutputGemmAvailable(), "OpenBLAS BGEMM is unavailable");

        Tensor a = tensor(DataType.BFLOAT16, 64, 64, "a");
        Tensor b = tensor(DataType.BFLOAT16, 64, 64, "b");
        Tensor out = a.matmul(b);

        var execution = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                .prepare(runtime(BlasStorageMode.CPU_NATIVE, 1L));
        var step = execution.forwardSteps().stream()
                .filter(candidate -> candidate.compiledNode().operation() != null
                        && candidate.compiledNode().operation().opType() == Operation.OpType.MATMUL)
                .findFirst()
                .orElseThrow();

        assertEquals(MatMulExecutionRoute.OPENBLAS_NATIVE_SEGMENT, step.metadata().cpuPlan().matMulHints().route());
        assertEquals("BF16NativeBlasMatMulExecutable", step.metadata().cpuPlan().matMulExecutable().getClass().getSimpleName());
    }

    @Test
    void nativeSegmentTraceReportsProviderSymbolAndCopyBytes() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor a = tensor(DataType.FLOAT32, 64, 64, "a");
        Tensor b = tensor(DataType.FLOAT32, 64, 64, "b");
        Tensor out = a.matmul(b);

        var trace = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                .executeTraced(runtime(BlasStorageMode.CPU_NATIVE, 1L), ExecutionMode.FORWARD);

        var matmul = trace.steps().stream()
                .filter(step -> "MATMUL".equals(step.opType()))
                .findFirst()
                .orElseThrow();

        assertNotNull(matmul.metadata().matMul());
        assertEquals("OPENBLAS_FFM", matmul.metadata().matMul().blasProvider());
        assertEquals("cblas_sgemm", matmul.metadata().matMul().blasSymbol());
        assertEquals("OPENBLAS_NATIVE_SEGMENT", matmul.metadata().matMul().route());
        assertEquals("AUTO_UNCONTROLLED", matmul.metadata().matMul().threadPolicy());
        assertEquals(64L * 64L * Float.BYTES * 2L, matmul.metadata().matMul().copyInBytes());
        assertEquals(0L, matmul.metadata().matMul().copyOutBytes());
        assertEquals(0L, matmul.metadata().matMul().nativeTempBytes());
        assertEquals("OPENBLAS_FFM", matmul.metadata().attributes().get("blasProvider"));
        assertEquals("cblas_sgemm", matmul.metadata().attributes().get("blasSymbol"));
        assertEquals("AUTO_UNCONTROLLED", matmul.metadata().attributes().get("blasThreadPolicy"));
    }

    private static graph.execution.PreparedNodeExecution matmulStep(DataType dataType, RuntimeConfig runtime) {
        Tensor a = tensor(dataType, 64, 64, "a");
        Tensor b = tensor(dataType, 64, 64, "b");
        Tensor out = a.matmul(b);
        return CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                .prepare(runtime)
                .forwardSteps().stream()
                .filter(step -> step.compiledNode().operation() != null
                        && step.compiledNode().operation().opType() == Operation.OpType.MATMUL)
                .findFirst()
                .orElseThrow();
    }

    private static RuntimeConfig runtime(BlasStorageMode storageMode, long minWork) {
        return new RuntimeConfig(
                KernelTuningConfig.defaultsInference(),
                ApproximationConfig.defaults(),
                new BlasConfig(
                        BlasProvider.OPENBLAS_FFM,
                        minWork,
                        false,
                        100.0d,
                        false,
                        100.0d,
                        storageMode,
                        false
                )
        );
    }

    private static Tensor tensor(DataType dataType, int rows, int cols, String label) {
        double[] values = new double[rows * cols];
        for (int i = 0; i < values.length; i++) {
            values[i] = Math.sin(i * 0.013);
        }
        return new Tensor(values, new int[]{rows, cols}, null, label, dataType);
    }
}
