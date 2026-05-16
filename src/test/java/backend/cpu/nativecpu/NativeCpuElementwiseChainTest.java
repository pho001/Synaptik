package backend.cpu.nativecpu;

import backend.blas.BlasProvider;
import backend.blas.OpenBlasFfmBridge;
import backend.runtime.ExecutionMode;
import config.backend.KernelTuningConfig;
import config.compile.CompileConfig;
import config.compile.RegionOptimizationConfig;
import config.compile.SemanticCanonicalizationConfig;
import config.runtime.ApproximationConfig;
import config.runtime.BlasConfig;
import config.runtime.BlasStorageMode;
import config.runtime.CpuStorageProfile;
import config.runtime.NativeCpuFailurePolicy;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import graph.execution.PublicationPolicy;
import graph.execution.trace.ExecutionStepTrace;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class NativeCpuElementwiseChainTest {
    @Test
    void cpuNativeMatmulReluKeepsReluOutputNative() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor out = a().matmul(b()).relu();

        var trace = CompiledGraph.compile(out, nativeElementwiseCompileConfig())
                .executeTraced(runtime(CpuStorageProfile.CPU_NATIVE, NativeCpuFailurePolicy.FALLBACK_TO_ARRAY), ExecutionMode.FORWARD, PublicationPolicy.NONE);

        Map<String, Object> relu = attrs(trace.steps().stream()
                .filter(step -> "RELU".equals(step.opType()))
                .findFirst()
                .orElseThrow());

        assertEquals("CPU_NATIVE", relu.get("cpuStorageProfile"));
        assertEquals("CPU_NATIVE", relu.get("requestedCpuStorage"));
        assertEquals("CPU_NATIVE", relu.get("actualCpuStorage"));
        assertEquals("NATIVE_CORRECT_BUT_SLOW", relu.get("nativeCpuKernelStatus"));
        assertEquals("SEGMENT_SCALAR", relu.get("nativeCpuKernelFamily"));
        assertEquals("", relu.get("nativeCpuFallbackReason"));
        assertEquals("CPU_NATIVE", relu.get("storageResidency"));
        assertEquals(false, relu.get("storageCpuCurrent"));
    }

    @Test
    void cpuNativeMatmulAddKeepsSameShapeAddOutputNative() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor c = tensor(new float[]{1f, -1f, 2f, -2f}, "c");
        Tensor out = a().matmul(b()).add(c);

        var trace = CompiledGraph.compile(out, nativeElementwiseCompileConfig())
                .executeTraced(runtime(CpuStorageProfile.CPU_NATIVE, NativeCpuFailurePolicy.FALLBACK_TO_ARRAY), ExecutionMode.FORWARD, PublicationPolicy.NONE);

        Map<String, Object> add = attrs(trace.steps().stream()
                .filter(step -> "ADD".equals(step.opType()))
                .findFirst()
                .orElseThrow());

        assertEquals("CPU_NATIVE", add.get("actualCpuStorage"));
        assertEquals("NATIVE_CORRECT_BUT_SLOW", add.get("nativeCpuKernelStatus"));
        assertEquals("SEGMENT_SCALAR", add.get("nativeCpuKernelFamily"));
        assertEquals("", add.get("nativeCpuFallbackReason"));
        assertEquals("CPU_NATIVE", add.get("storageResidency"));
    }

    @Test
    void cpuNativeMatmulBiasAddReluKeepsAddAndReluOutputNative() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor bias = vector(new float[]{1f, -100f}, "bias");
        Tensor out = a().matmul(b()).add(bias).relu();

        var trace = CompiledGraph.compile(out, nativeElementwiseCompileConfig())
                .executeTraced(runtime(CpuStorageProfile.CPU_NATIVE, NativeCpuFailurePolicy.FALLBACK_TO_ARRAY), ExecutionMode.FORWARD, PublicationPolicy.NONE);

        Map<String, Object> add = attrs(trace.steps().stream()
                .filter(step -> "ADD".equals(step.opType()))
                .findFirst()
                .orElseThrow());
        Map<String, Object> relu = attrs(trace.steps().stream()
                .filter(step -> "RELU".equals(step.opType()))
                .findFirst()
                .orElseThrow());

        assertEquals("CPU_NATIVE", add.get("actualCpuStorage"));
        assertEquals("NATIVE_CORRECT_BUT_SLOW", add.get("nativeCpuKernelStatus"));
        assertEquals("SEGMENT_SCALAR", add.get("nativeCpuKernelFamily"));
        assertEquals("", add.get("nativeCpuFallbackReason"));
        assertEquals("CPU_NATIVE", add.get("storageResidency"));
        assertEquals("CPU_NATIVE", relu.get("actualCpuStorage"));
        assertEquals("", relu.get("nativeCpuFallbackReason"));
        assertEquals("CPU_NATIVE", relu.get("storageResidency"));
    }

    @Test
    void cpuNativeMatmulBiasAddSupportsBiasOnLeftSide() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor bias = vector(new float[]{1f, -100f}, "bias");
        Tensor out = bias.add(a().matmul(b()));

        var trace = CompiledGraph.compile(out, nativeElementwiseCompileConfig())
                .executeTraced(runtime(CpuStorageProfile.CPU_NATIVE, NativeCpuFailurePolicy.FALLBACK_TO_ARRAY), ExecutionMode.FORWARD, PublicationPolicy.NONE);

        Map<String, Object> add = attrs(trace.steps().stream()
                .filter(step -> "ADD".equals(step.opType()))
                .findFirst()
                .orElseThrow());

        assertEquals("CPU_NATIVE", add.get("actualCpuStorage"));
        assertEquals("", add.get("nativeCpuFallbackReason"));
        assertEquals("CPU_NATIVE", add.get("storageResidency"));
    }

    @Test
    void cpuNativeMatmulMulReluKeepsMulAndReluOutputNative() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor scale = tensor(new float[]{2f, 3f, 4f, 5f}, "scale");
        Tensor out = a().matmul(b()).mul(scale).relu();

        var trace = CompiledGraph.compile(out, nativeElementwiseCompileConfig())
                .executeTraced(runtime(CpuStorageProfile.CPU_NATIVE, NativeCpuFailurePolicy.FALLBACK_TO_ARRAY), ExecutionMode.FORWARD, PublicationPolicy.NONE);

        Map<String, Object> mul = attrs(trace.steps().stream()
                .filter(step -> "MUL".equals(step.opType()))
                .findFirst()
                .orElseThrow());
        Map<String, Object> relu = attrs(trace.steps().stream()
                .filter(step -> "RELU".equals(step.opType()))
                .findFirst()
                .orElseThrow());

        assertEquals("CPU_NATIVE", mul.get("actualCpuStorage"));
        assertEquals("NATIVE_CORRECT_BUT_SLOW", mul.get("nativeCpuKernelStatus"));
        assertEquals("SEGMENT_SCALAR", mul.get("nativeCpuKernelFamily"));
        assertEquals("", mul.get("nativeCpuFallbackReason"));
        assertEquals("CPU_NATIVE", mul.get("storageResidency"));
        assertEquals("CPU_NATIVE", relu.get("actualCpuStorage"));
        assertEquals("", relu.get("nativeCpuFallbackReason"));
        assertEquals("CPU_NATIVE", relu.get("storageResidency"));
    }

    @Test
    void unsupportedCpuNativeBroadcastMulFallsBackToArrayWithTraceReason() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor scale = vector(new float[]{2f, 3f}, "scale");
        Tensor out = a().matmul(b()).mul(scale);

        var trace = CompiledGraph.compile(out, nativeElementwiseCompileConfig())
                .executeTraced(runtime(CpuStorageProfile.CPU_NATIVE, NativeCpuFailurePolicy.FALLBACK_TO_ARRAY), ExecutionMode.FORWARD, PublicationPolicy.NONE);

        Map<String, Object> mul = attrs(trace.steps().stream()
                .filter(step -> "MUL".equals(step.opType()))
                .findFirst()
                .orElseThrow());

        assertEquals("CPU_NATIVE", mul.get("requestedCpuStorage"));
        assertEquals("CPU_ARRAY", mul.get("actualCpuStorage"));
        assertEquals("native-kernel-ineligible:mul-broadcast", mul.get("nativeCpuFallbackReason"));
        assertEquals("CPU_ARRAY", mul.get("storageResidency"));
    }

    @Test
    void requireNativeRejectsUnsupportedCpuNativeBroadcastMul() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor scale = vector(new float[]{2f, 3f}, "scale");
        Tensor out = a().matmul(b()).mul(scale);

        IllegalStateException failure = assertThrows(
                IllegalStateException.class,
                () -> CompiledGraph.compile(out, nativeElementwiseCompileConfig())
                        .executeTraced(runtime(CpuStorageProfile.CPU_NATIVE, NativeCpuFailurePolicy.REQUIRE_NATIVE), ExecutionMode.FORWARD, PublicationPolicy.NONE)
        );

        assertTrue(failure.getMessage().contains("Native CPU execution required"));
        assertTrue(failure.getMessage().contains("native-kernel-ineligible:mul-broadcast"));
    }

    @Test
    void unsupportedCpuNativeBroadcastAddFallsBackToArrayWithTraceReason() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor columnBias = matrix(new float[]{1f, -100f}, new int[]{2, 1}, "column_bias");
        Tensor out = a().matmul(b()).add(columnBias);

        var trace = CompiledGraph.compile(out, nativeElementwiseCompileConfig())
                .executeTraced(runtime(CpuStorageProfile.CPU_NATIVE, NativeCpuFailurePolicy.FALLBACK_TO_ARRAY), ExecutionMode.FORWARD, PublicationPolicy.NONE);

        Map<String, Object> add = attrs(trace.steps().stream()
                .filter(step -> "ADD".equals(step.opType()))
                .findFirst()
                .orElseThrow());

        assertEquals("CPU_NATIVE", add.get("requestedCpuStorage"));
        assertEquals("CPU_ARRAY", add.get("actualCpuStorage"));
        assertEquals("native-kernel-ineligible:add-broadcast", add.get("nativeCpuFallbackReason"));
        assertEquals("CPU_ARRAY", add.get("storageResidency"));
    }

    @Test
    void requireNativeRejectsUnsupportedCpuNativeBroadcastAdd() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor columnBias = matrix(new float[]{1f, -100f}, new int[]{2, 1}, "column_bias");
        Tensor out = a().matmul(b()).add(columnBias);

        IllegalStateException failure = assertThrows(
                IllegalStateException.class,
                () -> CompiledGraph.compile(out, nativeElementwiseCompileConfig())
                        .executeTraced(runtime(CpuStorageProfile.CPU_NATIVE, NativeCpuFailurePolicy.REQUIRE_NATIVE), ExecutionMode.FORWARD, PublicationPolicy.NONE)
        );

        assertTrue(failure.getMessage().contains("Native CPU execution required"));
        assertTrue(failure.getMessage().contains("native-kernel-ineligible:add-broadcast"));
    }

    @Test
    void unsupportedCpuNativeElementwiseFallsBackToArrayWithTraceReason() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor out = a().matmul(b()).log();

        var trace = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                .executeTraced(runtime(CpuStorageProfile.CPU_NATIVE, NativeCpuFailurePolicy.FALLBACK_TO_ARRAY), ExecutionMode.FORWARD, PublicationPolicy.NONE);

        Map<String, Object> log = attrs(trace.steps().stream()
                .filter(step -> "LOG".equals(step.opType()))
                .findFirst()
                .orElseThrow());

        assertEquals("CPU_NATIVE", log.get("requestedCpuStorage"));
        assertEquals("CPU_ARRAY", log.get("actualCpuStorage"));
        assertEquals("NATIVE_UNSUPPORTED", log.get("nativeCpuKernelStatus"));
        assertEquals("ARRAY_ONLY", log.get("nativeCpuKernelFamily"));
        assertEquals("native-kernel-unsupported:log", log.get("nativeCpuFallbackReason"));
        assertEquals("CPU_ARRAY", log.get("storageResidency"));
    }

    @Test
    void requireNativeRejectsUnsupportedCpuNativeElementwise() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor out = a().matmul(b()).log();

        IllegalStateException failure = assertThrows(
                IllegalStateException.class,
                () -> CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                        .executeTraced(runtime(CpuStorageProfile.CPU_NATIVE, NativeCpuFailurePolicy.REQUIRE_NATIVE), ExecutionMode.FORWARD, PublicationPolicy.NONE)
        );

        assertTrue(failure.getMessage().contains("Native CPU execution required"));
        assertTrue(failure.getMessage().contains("native-kernel-unsupported:log"));
    }

    @Test
    void autoStorageDoesNotUseNativeElementwiseSlice() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor out = a().matmul(b()).add(vector(new float[]{1f, -100f}, "bias"));

        var trace = CompiledGraph.compile(out, nativeElementwiseCompileConfig())
                .executeTraced(runtime(CpuStorageProfile.AUTO, NativeCpuFailurePolicy.FALLBACK_TO_ARRAY), ExecutionMode.FORWARD, PublicationPolicy.NONE);

        Map<String, Object> add = attrs(trace.steps().stream()
                .filter(step -> "ADD".equals(step.opType()))
                .findFirst()
                .orElseThrow());

        assertFalse(add.containsKey("nativeCpuKernelStatus"));
        assertEquals("CPU_ARRAY", add.get("storageResidency"));
    }

    @Test
    void autoStorageDoesNotUseNativeMulSlice() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor out = a().matmul(b()).mul(tensor(new float[]{2f, 3f, 4f, 5f}, "scale"));

        var trace = CompiledGraph.compile(out, nativeElementwiseCompileConfig())
                .executeTraced(runtime(CpuStorageProfile.AUTO, NativeCpuFailurePolicy.FALLBACK_TO_ARRAY), ExecutionMode.FORWARD, PublicationPolicy.NONE);

        Map<String, Object> mul = attrs(trace.steps().stream()
                .filter(step -> "MUL".equals(step.opType()))
                .findFirst()
                .orElseThrow());

        assertFalse(mul.containsKey("nativeCpuKernelStatus"));
        assertEquals("CPU_ARRAY", mul.get("storageResidency"));
    }

    private static RuntimeConfig runtime(CpuStorageProfile storageProfile, NativeCpuFailurePolicy failurePolicy) {
        return new RuntimeConfig(
                KernelTuningConfig.defaultsInference(),
                ApproximationConfig.defaults(),
                new BlasConfig(
                        BlasProvider.OPENBLAS_FFM,
                        1L,
                        false,
                        100.0d,
                        false,
                        100.0d,
                        BlasStorageMode.AUTO,
                        false
                )
        )
                .withCpuStorageProfile(storageProfile)
                .withNativeCpuFailurePolicy(failurePolicy);
    }

    private static CompileConfig nativeElementwiseCompileConfig() {
        return CompileConfig.noGraphOptimizationBaseline()
                .withSemanticCanonicalization(SemanticCanonicalizationConfig.disabled())
                .withRegionOptimization(RegionOptimizationConfig.disabled());
    }

    private static Tensor a() {
        return tensor(new float[]{1f, 2f, 3f, 4f}, "a");
    }

    private static Tensor b() {
        return tensor(new float[]{5f, 6f, 7f, 8f}, "b");
    }

    private static Tensor tensor(float[] values, String label) {
        return new Tensor(values, new int[]{2, 2}, null, label, DataType.FLOAT32);
    }

    private static Tensor vector(float[] values, String label) {
        return new Tensor(values, new int[]{values.length}, null, label, DataType.FLOAT32);
    }

    private static Tensor matrix(float[] values, int[] shape, String label) {
        return new Tensor(values, shape, null, label, DataType.FLOAT32);
    }

    private static Map<String, Object> attrs(ExecutionStepTrace step) {
        return step.metadata().attributes();
    }
}
