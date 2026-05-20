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
import config.runtime.CpuStorageProfile;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import operations.Operation;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

class NativeOpenBlasPlannerTest {
    @Test
    void cpuArrayProfileOverridesAutoStorageModeWithArrayCopyRoute() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isAvailable(), OpenBlasFfmBridge.unavailableReason());

        var step = matmulStep(DataType.FLOAT32, runtime(CpuStorageProfile.CPU_ARRAY, BlasStorageMode.AUTO, 1L));

        assertEquals(MatMulExecutionRoute.OPENBLAS_ARRAY_COPYING, testsupport.MetadataArtifacts.cpuPlan(step.metadata()).matMulHints().route());
        assertEquals("F32BlasMatMulExecutable", testsupport.MetadataArtifacts.cpuPlan(step.metadata()).matMulExecutable().getClass().getSimpleName());
    }

    @Test
    void cpuArrayProfileOverridesNativeStorageModeWithArrayCopyRoute() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isAvailable(), OpenBlasFfmBridge.unavailableReason());

        var step = matmulStep(DataType.FLOAT32, runtime(CpuStorageProfile.CPU_ARRAY, BlasStorageMode.CPU_NATIVE, 1L));

        assertEquals(MatMulExecutionRoute.OPENBLAS_ARRAY_COPYING, testsupport.MetadataArtifacts.cpuPlan(step.metadata()).matMulHints().route());
        assertEquals("F32BlasMatMulExecutable", testsupport.MetadataArtifacts.cpuPlan(step.metadata()).matMulExecutable().getClass().getSimpleName());
    }

    @Test
    void cpuNativeProfileSelectsFloat32NativeSegmentRouteEvenWhenBlasStorageModeIsAuto() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        var step = matmulStep(DataType.FLOAT32, runtime(CpuStorageProfile.CPU_NATIVE, BlasStorageMode.AUTO, Long.MAX_VALUE));

        assertEquals(MatMulExecutionRoute.OPENBLAS_NATIVE_SEGMENT, testsupport.MetadataArtifacts.cpuPlan(step.metadata()).matMulHints().route());
        assertEquals("F32NativeBlasMatMulExecutable", testsupport.MetadataArtifacts.cpuPlan(step.metadata()).matMulExecutable().getClass().getSimpleName());
    }

    @Test
    void autoStorageModeSelectsFloat64NativeSegmentRouteForEligibleDenseMatmul() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat64GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        var step = matmulStep(DataType.FLOAT64, runtime(CpuStorageProfile.AUTO, BlasStorageMode.AUTO, 1L));

        assertEquals(MatMulExecutionRoute.OPENBLAS_NATIVE_SEGMENT, testsupport.MetadataArtifacts.cpuPlan(step.metadata()).matMulHints().route());
        assertEquals("F64NativeBlasMatMulExecutable", testsupport.MetadataArtifacts.cpuPlan(step.metadata()).matMulExecutable().getClass().getSimpleName());
    }

    @Test
    void cpuNativeProfileDoesNotEnableBlasWhenProviderIsDisabled() {
        Tensor a = tensor(DataType.FLOAT32, 64, 64, "a");
        Tensor b = tensor(DataType.FLOAT32, 64, 64, "b");
        Tensor out = a.matmul(b);

        RuntimeConfig runtime = RuntimeConfig.inferenceDefaults()
                .withCpuStorageProfile(CpuStorageProfile.CPU_NATIVE);
        var step = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                .prepare(runtime)
                .forwardSteps().stream()
                .filter(candidate -> candidate.compiledNode().operation() != null
                        && candidate.compiledNode().operation().opType() == Operation.OpType.MATMUL)
                .findFirst()
                .orElseThrow();

        assertEquals(MatMulExecutionRoute.JAVA_DIRECT, testsupport.MetadataArtifacts.cpuPlan(step.metadata()).matMulHints().route());
        assertEquals("F32JavaMatMulExecutable", testsupport.MetadataArtifacts.cpuPlan(step.metadata()).matMulExecutable().getClass().getSimpleName());
    }

    @Test
    void bfloat16SelectsNativeSegmentRouteWhenBgemmIsAvailable() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isBFloat16OutputGemmAvailable(), "OpenBLAS BGEMM is unavailable");

        Tensor a = tensor(DataType.BFLOAT16, 64, 64, "a");
        Tensor b = tensor(DataType.BFLOAT16, 64, 64, "b");
        Tensor out = a.matmul(b);

        var execution = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                .prepare(runtime(CpuStorageProfile.CPU_NATIVE, BlasStorageMode.AUTO, 1L));
        var step = execution.forwardSteps().stream()
                .filter(candidate -> candidate.compiledNode().operation() != null
                        && candidate.compiledNode().operation().opType() == Operation.OpType.MATMUL)
                .findFirst()
                .orElseThrow();

        assertEquals(MatMulExecutionRoute.OPENBLAS_NATIVE_SEGMENT, testsupport.MetadataArtifacts.cpuPlan(step.metadata()).matMulHints().route());
        assertEquals("BF16NativeBlasMatMulExecutable", testsupport.MetadataArtifacts.cpuPlan(step.metadata()).matMulExecutable().getClass().getSimpleName());
    }

    @Test
    void nativeSegmentTraceReportsProviderSymbolAndCopyBytes() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor a = tensor(DataType.FLOAT32, 64, 64, "a");
        Tensor b = tensor(DataType.FLOAT32, 64, 64, "b");
        Tensor out = a.matmul(b);

        var trace = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                .prepare(runtime(CpuStorageProfile.CPU_NATIVE, BlasStorageMode.AUTO, 1L)).executeTraced(ExecutionMode.FORWARD);

        var matmul = trace.steps().stream()
                .filter(step -> "MATMUL".equals(step.opType()))
                .findFirst()
                .orElseThrow();

        assertNotNull(matmul.metadata().matMul());
        assertEquals("OPENBLAS_FFM", matmul.metadata().matMul().blasProvider());
        assertEquals("cblas_sgemm", matmul.metadata().matMul().blasSymbol());
        assertEquals("OPENBLAS_NATIVE_SEGMENT", matmul.metadata().matMul().route());
        assertEquals("CPU_NATIVE", matmul.metadata().matMul().cpuStorageProfile());
        assertEquals("FALLBACK_TO_ARRAY", matmul.metadata().matMul().nativeCpuFailurePolicy());
        assertEquals("CPU_NATIVE", matmul.metadata().matMul().requestedCpuStorage());
        assertEquals("CPU_NATIVE", matmul.metadata().matMul().actualCpuStorage());
        assertEquals("", matmul.metadata().matMul().nativeCpuFallbackReason());
        assertEquals("AUTO_UNCONTROLLED", matmul.metadata().matMul().threadPolicy());
        assertEquals(64L * 64L * Float.BYTES * 2L, matmul.metadata().matMul().copyInBytes());
        assertEquals(0L, matmul.metadata().matMul().copyOutBytes());
        assertEquals(0L, matmul.metadata().matMul().nativeTempBytes());
        assertEquals("OPENBLAS_FFM", matmul.metadata().attributes().get("blasProvider"));
        assertEquals("cblas_sgemm", matmul.metadata().attributes().get("blasSymbol"));
        assertEquals("CPU_NATIVE", matmul.metadata().attributes().get("cpuStorageProfile"));
        assertEquals("FALLBACK_TO_ARRAY", matmul.metadata().attributes().get("nativeCpuFailurePolicy"));
        assertEquals("CPU_NATIVE", matmul.metadata().attributes().get("requestedCpuStorage"));
        assertEquals("CPU_NATIVE", matmul.metadata().attributes().get("actualCpuStorage"));
        assertEquals("", matmul.metadata().attributes().get("nativeCpuFallbackReason"));
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

    private static RuntimeConfig runtime(CpuStorageProfile cpuStorageProfile, BlasStorageMode storageMode, long minWork) {
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
        ).withCpuStorageProfile(cpuStorageProfile);
    }

    private static Tensor tensor(DataType dataType, int rows, int cols, String label) {
        double[] values = new double[rows * cols];
        for (int i = 0; i < values.length; i++) {
            values[i] = Math.sin(i * 0.013);
        }
        return new Tensor(values, new int[]{rows, cols}, null, label, dataType);
    }
}
