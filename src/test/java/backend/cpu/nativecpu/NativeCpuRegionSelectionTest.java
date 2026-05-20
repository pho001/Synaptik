package backend.cpu.nativecpu;

import backend.blas.BlasProvider;
import backend.runtime.ExecutionMode;
import config.backend.KernelTuningConfig;
import config.compile.CompileConfig;
import config.compile.SemanticCanonicalizationConfig;
import config.runtime.ApproximationConfig;
import config.runtime.BlasConfig;
import config.runtime.BlasStorageMode;
import config.runtime.CpuStorageProfile;
import config.runtime.NativeCpuFailurePolicy;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import graph.execution.PublicationPolicy;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class NativeCpuRegionSelectionTest {
    @Test
    void preparedExecutionNoLongerEmitsNativeCpuChainTraceAttrs() {
        Tensor left = new Tensor(new float[]{1f, -2f, 3f, -4f}, new int[]{2, 2}, null, "left", DataType.FLOAT32);
        Tensor right = new Tensor(new float[]{5f, 6f, 7f, 8f}, new int[]{2, 2}, null, "right", DataType.FLOAT32);
        Tensor out = left.add(right).relu();

        var trace = CompiledGraph.compile(out, CompileConfig.inference())
                .prepare(
                        RuntimeConfig.inferenceDefaults()
                                .withCpuStorageProfile(CpuStorageProfile.CPU_NATIVE)
                                .withNativeCpuFailurePolicy(NativeCpuFailurePolicy.FALLBACK_TO_ARRAY)).executeTraced(ExecutionMode.FORWARD, PublicationPolicy.NONE);

        assertFalse(trace.steps().stream()
                .flatMap(step -> step.metadata().attributes().keySet().stream())
                .anyMatch(key -> key.startsWith("nativeCpuChain")));
    }

    @Test
    void cpuNativeWithDisabledBlasEmitsNativeRegionProviderUnavailableRejection() {
        Tensor left = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "left", DataType.FLOAT32);
        Tensor right = new Tensor(new float[]{5f, 6f, 7f, 8f}, new int[]{2, 2}, null, "right", DataType.FLOAT32);
        Tensor out = left.matmul(right).relu();

        var trace = CompiledGraph.compile(out, noSemanticLinearFusion())
                .prepare(
                        disabledBlasRuntime(CpuStorageProfile.CPU_NATIVE)).executeTraced(ExecutionMode.FORWARD, PublicationPolicy.NONE);

        Map<String, Object> matmulAttrs = trace.steps().stream()
                .filter(step -> "MATMUL".equals(step.opType()))
                .map(step -> step.metadata().attributes())
                .findFirst()
                .orElseThrow();

        assertEquals("REJECTED", matmulAttrs.get("nativeCpuRegionDecision"));
        assertEquals("CPU_ARRAY", matmulAttrs.get("nativeCpuRegionRoute"));
        assertEquals("native-cpu-region-provider-unavailable:matmul", matmulAttrs.get("nativeCpuRegionReason"));
        assertEquals("native-cpu-region-provider-unavailable:matmul", matmulAttrs.get("nativeCpuRegionFallbackReason"));
        assertEquals(1, matmulAttrs.get("nativeCpuRegionNodeCount"));
        assertEquals("MATMUL", matmulAttrs.get("nativeCpuRegionRejectedOp"));
        assertEquals(List.of(true), matmulAttrs.get("nativeCpuParityAutoEligible"));
        assertNestedContains(matmulAttrs, "nativeCpuParityStoragePaths", "CPU_NATIVE_REGION_PROVIDER");
        assertNestedContains(matmulAttrs, "nativeCpuParityResultResidencies", "CPU_NATIVE");
        assertTrue(trace.steps().stream()
                .noneMatch(step -> "SELECTED".equals(step.metadata().attributes().get("nativeCpuRegionDecision"))));
    }

    @Test
    void cpuNativeProviderRejectsStridedInputWithLayoutMaterializationReason() {
        Tensor leftBase = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "leftBase", DataType.FLOAT32);
        Tensor left = leftBase.transpose();
        Tensor right = new Tensor(new float[]{7f, 8f, 9f, 10f}, new int[]{2, 2}, null, "right", DataType.FLOAT32);
        Tensor out = left.matmul(right).relu();

        var trace = CompiledGraph.compile(out, noSemanticLinearFusion())
                .prepare(
                        disabledBlasRuntime(CpuStorageProfile.CPU_NATIVE)).executeTraced(ExecutionMode.FORWARD, PublicationPolicy.NONE);

        Map<String, Object> matmulAttrs = trace.steps().stream()
                .filter(step -> "MATMUL".equals(step.opType()))
                .map(step -> step.metadata().attributes())
                .findFirst()
                .orElseThrow();

        assertEquals("REJECTED", matmulAttrs.get("nativeCpuRegionDecision"));
        assertEquals("native-layout-materialization-required:provider-dense-input",
                matmulAttrs.get("nativeCpuRegionReason"));
        assertEquals(1, matmulAttrs.get("nativeCpuStridedNodeCount"));
        assertEquals(1, matmulAttrs.get("nativeCpuStridedMaterializationCount"));
        assertEquals(List.of("native-layout-materialization-required:provider-dense-input"),
                matmulAttrs.get("nativeCpuStridedFallbackReasons"));
        assertEquals(List.of(true), matmulAttrs.get("nativeCpuParityAutoEligible"));
        assertNestedContains(matmulAttrs, "nativeCpuParityStoragePaths", "CPU_NATIVE_REGION_PROVIDER");
        assertNestedContains(matmulAttrs, "nativeCpuParityLayoutCapabilities", "DENSE");
        @SuppressWarnings("unchecked")
        Map<String, Integer> layoutClassCounts =
                (Map<String, Integer>) matmulAttrs.get("nativeCpuLayoutClassCounts");
        assertEquals(2, layoutClassCounts.get("DENSE_CONTIGUOUS"));
        assertEquals(1, layoutClassCounts.get("TRANSPOSE_2D_READ_DENSE_WRITE"));
    }

    private static RuntimeConfig disabledBlasRuntime(CpuStorageProfile profile) {
        return new RuntimeConfig(
                KernelTuningConfig.defaultsInference(),
                ApproximationConfig.defaults(),
                new BlasConfig(
                        BlasProvider.NONE,
                        1L,
                        false,
                        100.0d,
                        false,
                        100.0d,
                        BlasStorageMode.AUTO,
                        false
                )
        )
                .withCpuStorageProfile(profile)
                .withNativeCpuFailurePolicy(NativeCpuFailurePolicy.FALLBACK_TO_ARRAY);
    }

    private static CompileConfig noSemanticLinearFusion() {
        return CompileConfig.inference()
                .withSemanticCanonicalization(SemanticCanonicalizationConfig.disabled());
    }

    private static void assertNestedContains(Map<String, Object> attrs, String key, String expected) {
        Object value = attrs.get(key);
        assertTrue(value instanceof List<?>, key + " should be a nested list");
        @SuppressWarnings("unchecked")
        List<List<String>> rows = (List<List<String>>) value;
        assertTrue(rows.stream().anyMatch(row -> row.contains(expected)), key + " missing " + expected + ": " + rows);
    }
}
