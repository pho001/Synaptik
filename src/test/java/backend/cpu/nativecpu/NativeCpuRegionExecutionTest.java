package backend.cpu.nativecpu;

import backend.ComputeBackend;
import backend.accelerator.exec.PartitionExecutionRole;
import backend.blas.BlasProvider;
import backend.blas.OpenBlasFfmBridge;
import backend.lowering.LoweringFamily;
import backend.memory.CpuMaterializationReason;
import backend.runtime.ExecutionMode;
import config.backend.KernelTuningConfig;
import config.compile.CompileConfig;
import config.compile.SemanticCanonicalizationConfig;
import config.runtime.ApproximationConfig;
import config.runtime.BlasConfig;
import config.runtime.BlasStorageMode;
import config.runtime.CpuStorageProfile;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import graph.execution.PublicationPolicy;
import graph.execution.trace.ExecutionStepTrace;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class NativeCpuRegionExecutionTest {
    @Test
    void cpuNativeMatmulPreparesSingleNativeRegionAnchor() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor a = tensor("a");
        Tensor b = tensor("b");
        Tensor out = a.matmul(b).relu();

        var prepared = CompiledGraph.compile(out, CompileConfig.inference()).prepare(openBlasRuntime(CpuStorageProfile.CPU_NATIVE));
        var regionStep = prepared.forwardSteps().stream()
                .filter(step -> step.metadata().cpuRegionExecutable() != null)
                .findFirst()
                .orElseThrow();

        assertEquals(ComputeBackend.CPU, regionStep.metadata().backend());
        assertEquals(PartitionExecutionRole.ANCHOR, regionStep.metadata().partitionRole());
        assertEquals(LoweringFamily.CPU_NATIVE_REGION, regionStep.metadata().cpuRegionExecutable().regionExecutionPlan().loweringFamily());
        assertEquals(List.of(regionStep.compiledNode().id()), regionStep.metadata().cpuRegionExecutable().regionExecutionPlan().boundaryOutputNodeIds());
    }

    @Test
    void cpuNativeRegionTraceReportsRegionAttrs() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor a = tensor("a");
        Tensor b = tensor("b");
        Tensor out = a.matmul(b).relu();

        var trace = CompiledGraph.compile(out, CompileConfig.inference())
                .executeTraced(openBlasRuntime(CpuStorageProfile.CPU_NATIVE), ExecutionMode.FORWARD);
        var step = trace.steps().stream()
                .filter(candidate -> candidate.metadata().attributes().containsKey("nativeCpuRegionId"))
                .findFirst()
                .orElseThrow();

        assertNotNull(step.metadata().attributes().get("regionId"));
        assertEquals("CPU_NATIVE_REGION", step.metadata().attributes().get("loweringFamily"));
        assertEquals("SELECTED", step.metadata().attributes().get("nativeCpuRegionDecision"));
        assertEquals("NATIVE", step.metadata().attributes().get("nativeCpuRegionRoute"));
        assertEquals("", step.metadata().attributes().get("nativeCpuRegionFallbackReason"));
        assertEquals(1, step.metadata().attributes().get("nativeCpuRegionLocalKernelCount"));
        assertEquals(2, step.metadata().attributes().get("nativeCpuRegionExecutedGroupCount"));
        assertEquals("OPENBLAS_FFM", step.metadata().attributes().get("nativeCpuRegionProviderKind"));
        assertEquals(1, ((List<?>) step.metadata().attributes().get("nativeCpuRegionProviderNodes")).size());
        assertEquals(1, ((List<?>) step.metadata().attributes().get("nativeCpuRegionLocalKernelNodes")).size());
        assertEquals(1, step.metadata().attributes().get("nativeCpuRegionFallbackPlanCount"));
        assertEquals(0L, step.metadata().attributes().get("nativeCpuStridedNodeCount"));
        assertEquals(0L, step.metadata().attributes().get("nativeCpuStridedMaterializationCount"));
        assertEquals(List.of(), step.metadata().attributes().get("nativeCpuStridedFallbackReasons"));
        assertEquals(List.of("PROVIDER", "SEGMENT_DENSE_SCALAR"),
                step.metadata().attributes().get("nativeCpuRegionSegmentKernelFamilies"));
        assertEquals(List.of(true, false), step.metadata().attributes().get("nativeCpuParityAutoEligible"));
        @SuppressWarnings("unchecked")
        List<List<String>> parityStoragePaths =
                (List<List<String>>) step.metadata().attributes().get("nativeCpuParityStoragePaths");
        assertTrue(parityStoragePaths.get(0).contains("CPU_NATIVE_REGION_PROVIDER"));
        assertTrue(parityStoragePaths.get(1).contains("CPU_NATIVE_REGION_DENSE"));
        @SuppressWarnings("unchecked")
        List<List<String>> parityLayoutCapabilities =
                (List<List<String>>) step.metadata().attributes().get("nativeCpuParityLayoutCapabilities");
        assertTrue(parityLayoutCapabilities.get(1).contains("DENSE"));
        assertTrue(parityLayoutCapabilities.get(1).contains("OFFSET_CONTIGUOUS"));
        @SuppressWarnings("unchecked")
        List<List<String>> parityResultResidencies =
                (List<List<String>>) step.metadata().attributes().get("nativeCpuParityResultResidencies");
        assertEquals(List.of("CPU_NATIVE"), parityResultResidencies.get(0));
        assertEquals(List.of("CPU_NATIVE"), parityResultResidencies.get(1));
        @SuppressWarnings("unchecked")
        Map<String, Integer> layoutClassCounts =
                (Map<String, Integer>) step.metadata().attributes().get("nativeCpuLayoutClassCounts");
        assertEquals(2, layoutClassCounts.get("DENSE_CONTIGUOUS"));
    }

    @Test
    void cpuNativeRegionExecutesBiasAddReluAsRegionLocalKernels() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor cpuA = tensor("cpuA");
        Tensor cpuB = tensor("cpuB");
        Tensor cpuBias = bias("cpuBias");
        Tensor cpuOut = cpuA.matmul(cpuB).add(cpuBias).relu();
        CompiledGraph.compile(cpuOut, noSemanticLinearFusion())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor a = tensor("a");
        Tensor b = tensor("b");
        Tensor bias = bias("bias");
        Tensor out = a.matmul(b).add(bias).relu();

        var trace = CompiledGraph.compile(out, noSemanticLinearFusion())
                .executeTraced(
                        openBlasRuntime(CpuStorageProfile.CPU_NATIVE),
                        ExecutionMode.FORWARD,
                        PublicationPolicy.NONE
                );
        var step = trace.steps().stream()
                .filter(candidate -> candidate.metadata().attributes().containsKey("nativeCpuRegionId"))
                .findFirst()
                .orElseThrow();

        assertEquals("NATIVE", step.metadata().attributes().get("nativeCpuRegionRoute"));
        assertEquals(2, step.metadata().attributes().get("nativeCpuRegionLocalKernelCount"));
        assertEquals(2, step.metadata().attributes().get("nativeCpuRegionExecutedGroupCount"));

        CompiledGraph.compile(out, noSemanticLinearFusion())
                .execute(openBlasRuntime(CpuStorageProfile.CPU_NATIVE), ExecutionMode.FORWARD);
        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 1e-4);
    }

    @Test
    void cpuNativeRegionExecutesReshapeAsRegionLocalViewAlias() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor cpuA = tensor("cpuA");
        Tensor cpuB = tensor("cpuB");
        Tensor cpuOut = cpuA.matmul(cpuB).reshape(32, 128).relu();
        CompiledGraph.compile(cpuOut, noSemanticLinearFusion())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor a = tensor("a");
        Tensor b = tensor("b");
        Tensor out = a.matmul(b).reshape(32, 128).relu();

        var trace = CompiledGraph.compile(out, noSemanticLinearFusion())
                .executeTraced(
                        openBlasRuntime(CpuStorageProfile.CPU_NATIVE),
                        ExecutionMode.FORWARD,
                        PublicationPolicy.NONE
                );
        var step = trace.steps().stream()
                .filter(candidate -> candidate.metadata().attributes().containsKey("nativeCpuRegionId"))
                .findFirst()
                .orElseThrow();

        assertEquals("NATIVE", step.metadata().attributes().get("nativeCpuRegionRoute"));
        assertEquals(1, step.metadata().attributes().get("nativeCpuRegionLocalKernelCount"));
        assertEquals(1, step.metadata().attributes().get("nativeCpuRegionLocalViewCount"));
        assertEquals(3, step.metadata().attributes().get("nativeCpuRegionExecutedGroupCount"));
        assertEquals(1, ((List<?>) step.metadata().attributes().get("nativeCpuRegionViewNodes")).size());

        CompiledGraph.compile(out, noSemanticLinearFusion())
                .execute(openBlasRuntime(CpuStorageProfile.CPU_NATIVE), ExecutionMode.FORWARD);
        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 1e-4);
    }

    @Test
    void cpuNativeRegionExecutesPermuteReluWithSegmentStridedKernel() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor cpuA = tensor("cpuA");
        Tensor cpuB = tensor("cpuB");
        Tensor cpuOut = cpuA.matmul(cpuB).transpose().relu();
        CompiledGraph.compile(cpuOut, noSemanticLinearFusion())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor a = tensor("a");
        Tensor b = tensor("b");
        Tensor out = a.matmul(b).transpose().relu();

        var trace = CompiledGraph.compile(out, noSemanticLinearFusion())
                .executeTraced(
                        openBlasRuntime(CpuStorageProfile.CPU_NATIVE),
                        ExecutionMode.FORWARD,
                        PublicationPolicy.NONE
                );
        var step = trace.steps().stream()
                .filter(candidate -> candidate.metadata().attributes().containsKey("nativeCpuRegionId"))
                .findFirst()
                .orElseThrow();

        assertEquals("NATIVE", step.metadata().attributes().get("nativeCpuRegionRoute"));
        assertEquals("", step.metadata().attributes().get("nativeCpuRegionFallbackReason"));
        assertEquals(1, step.metadata().attributes().get("nativeCpuRegionLocalKernelCount"));
        assertEquals(1, step.metadata().attributes().get("nativeCpuRegionLocalViewCount"));
        assertEquals(3, step.metadata().attributes().get("nativeCpuRegionExecutedGroupCount"));
        assertEquals(1L, step.metadata().attributes().get("nativeCpuStridedNodeCount"));
        assertEquals(0L, step.metadata().attributes().get("nativeCpuStridedMaterializationCount"));
        assertEquals(List.of(), step.metadata().attributes().get("nativeCpuStridedFallbackReasons"));
        assertTrue(((List<?>) step.metadata().attributes().get("nativeCpuRegionSegmentKernelFamilies"))
                .contains("SEGMENT_STRIDED_SCALAR"));

        CompiledGraph.compile(out, noSemanticLinearFusion())
                .execute(openBlasRuntime(CpuStorageProfile.CPU_NATIVE), ExecutionMode.FORWARD);
        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 1e-4);
    }

    @Test
    void cpuNativeRegionExecutesSelectAndSliceViewsWithSegmentConsumers() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor cpuA = tensor("cpuA");
        Tensor cpuB = tensor("cpuB");
        Tensor cpuSelected = cpuA.matmul(cpuB).select(0, 1).relu();
        Tensor cpuSliced = cpuA.matmul(cpuB).slice(new int[]{1, 0}, new int[]{3, 64}, new int[]{0, 1}, new int[]{1, 1}).relu();
        CompiledGraph.compile(cpuSelected, noSemanticLinearFusion())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);
        CompiledGraph.compile(cpuSliced, noSemanticLinearFusion())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor a = tensor("a");
        Tensor b = tensor("b");
        Tensor selected = a.matmul(b).select(0, 1).relu();
        Tensor sliced = a.matmul(b).slice(new int[]{1, 0}, new int[]{3, 64}, new int[]{0, 1}, new int[]{1, 1}).relu();

        var selectTrace = CompiledGraph.compile(selected, noSemanticLinearFusion())
                .executeTraced(
                        openBlasRuntime(CpuStorageProfile.CPU_NATIVE),
                        ExecutionMode.FORWARD,
                        PublicationPolicy.NONE
                );
        var sliceTrace = CompiledGraph.compile(sliced, noSemanticLinearFusion())
                .executeTraced(
                        openBlasRuntime(CpuStorageProfile.CPU_NATIVE),
                        ExecutionMode.FORWARD,
                        PublicationPolicy.NONE
                );
        var selectStep = nativeRegionStep(selectTrace.steps());
        var sliceStep = nativeRegionStep(sliceTrace.steps());

        assertEquals("NATIVE", selectStep.metadata().attributes().get("nativeCpuRegionRoute"));
        assertEquals(1, selectStep.metadata().attributes().get("nativeCpuRegionLocalKernelCount"));
        assertEquals(1, selectStep.metadata().attributes().get("nativeCpuRegionLocalViewCount"));
        assertEquals(1L, selectStep.metadata().attributes().get("nativeCpuStridedNodeCount"));
        assertTrue(((List<?>) selectStep.metadata().attributes().get("nativeCpuRegionSegmentKernelFamilies"))
                .contains("VIEW_ALIAS"));
        assertTrue(((List<?>) selectStep.metadata().attributes().get("nativeCpuRegionSegmentKernelFamilies"))
                .contains("SEGMENT_STRIDED_SCALAR"));

        assertEquals("NATIVE", sliceStep.metadata().attributes().get("nativeCpuRegionRoute"));
        assertEquals(1, sliceStep.metadata().attributes().get("nativeCpuRegionLocalKernelCount"));
        assertEquals(1, sliceStep.metadata().attributes().get("nativeCpuRegionLocalViewCount"));
        assertEquals(1L, sliceStep.metadata().attributes().get("nativeCpuStridedNodeCount"));

        CompiledGraph.compile(selected, noSemanticLinearFusion())
                .execute(openBlasRuntime(CpuStorageProfile.CPU_NATIVE), ExecutionMode.FORWARD);
        CompiledGraph.compile(sliced, noSemanticLinearFusion())
                .execute(openBlasRuntime(CpuStorageProfile.CPU_NATIVE), ExecutionMode.FORWARD);
        assertArrayEquals(cpuSelected.toDoubleArrayCopy(), selected.toDoubleArrayCopy(), 1e-4);
        assertArrayEquals(cpuSliced.toDoubleArrayCopy(), sliced.toDoubleArrayCopy(), 1e-4);
    }

    @Test
    void cpuNativeRegionExecutesExpandViewAsZeroStrideSegmentInput() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor cpuA = tensor("cpuA");
        Tensor cpuB = tensor("cpuB");
        Tensor cpuMatmul = cpuA.matmul(cpuB);
        Tensor cpuOut = cpuMatmul.add(cpuMatmul.select(0, 0).expand(64, 64));
        CompiledGraph.compile(cpuOut, noSemanticLinearFusion())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor a = tensor("a");
        Tensor b = tensor("b");
        Tensor matmul = a.matmul(b);
        Tensor out = matmul.add(matmul.select(0, 0).expand(64, 64));

        var trace = CompiledGraph.compile(out, noSemanticLinearFusion())
                .executeTraced(
                        openBlasRuntime(CpuStorageProfile.CPU_NATIVE),
                        ExecutionMode.FORWARD,
                        PublicationPolicy.NONE
                );
        var step = nativeRegionStep(trace.steps());

        assertEquals("NATIVE", step.metadata().attributes().get("nativeCpuRegionRoute"));
        assertEquals(1, step.metadata().attributes().get("nativeCpuRegionLocalKernelCount"));
        assertEquals(2, step.metadata().attributes().get("nativeCpuRegionLocalViewCount"));
        assertEquals(1L, step.metadata().attributes().get("nativeCpuStridedNodeCount"));
        @SuppressWarnings("unchecked")
        Map<String, Integer> layoutClassCounts =
                (Map<String, Integer>) step.metadata().attributes().get("nativeCpuLayoutClassCounts");
        assertTrue(layoutClassCounts.getOrDefault("BROADCAST_READ_DENSE_WRITE", 0)
                + layoutClassCounts.getOrDefault("LAST_DIM_BIAS_BROADCAST", 0) >= 1);
        assertTrue(((List<?>) step.metadata().attributes().get("nativeCpuRegionSegmentKernelFamilies"))
                .contains("VIEW_ALIAS"));
        assertTrue(((List<?>) step.metadata().attributes().get("nativeCpuRegionSegmentKernelFamilies"))
                .contains("SEGMENT_STRIDED_SCALAR"));

        CompiledGraph.compile(out, noSemanticLinearFusion())
                .execute(openBlasRuntime(CpuStorageProfile.CPU_NATIVE), ExecutionMode.FORWARD);
        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 1e-4);
    }

    @Test
    void cpuNativeRegionExecutesStridedBroadcastBinaryWithSegmentKernel() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor cpuA = tensor("cpuA");
        Tensor cpuB = tensor("cpuB");
        Tensor cpuBias = bias("cpuBias");
        Tensor cpuOut = cpuA.matmul(cpuB).transpose().add(cpuBias);
        CompiledGraph.compile(cpuOut, noSemanticLinearFusion())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor a = tensor("a");
        Tensor b = tensor("b");
        Tensor bias = bias("bias");
        Tensor out = a.matmul(b).transpose().add(bias);

        var trace = CompiledGraph.compile(out, noSemanticLinearFusion())
                .executeTraced(
                        openBlasRuntime(CpuStorageProfile.CPU_NATIVE),
                        ExecutionMode.FORWARD,
                        PublicationPolicy.NONE
                );
        var step = trace.steps().stream()
                .filter(candidate -> candidate.metadata().attributes().containsKey("nativeCpuRegionId"))
                .findFirst()
                .orElseThrow();

        assertEquals("NATIVE", step.metadata().attributes().get("nativeCpuRegionRoute"));
        assertEquals("", step.metadata().attributes().get("nativeCpuRegionFallbackReason"));
        assertEquals(1, step.metadata().attributes().get("nativeCpuRegionLocalKernelCount"));
        assertEquals(1, step.metadata().attributes().get("nativeCpuRegionLocalViewCount"));
        assertEquals(3, step.metadata().attributes().get("nativeCpuRegionExecutedGroupCount"));
        assertEquals(1L, step.metadata().attributes().get("nativeCpuStridedNodeCount"));
        assertTrue(((List<?>) step.metadata().attributes().get("nativeCpuRegionSegmentKernelFamilies"))
                .contains("SEGMENT_STRIDED_SCALAR"));

        CompiledGraph.compile(out, noSemanticLinearFusion())
                .execute(openBlasRuntime(CpuStorageProfile.CPU_NATIVE), ExecutionMode.FORWARD);
        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 1e-4);
    }

    @Test
    void cpuNativeRegionExecutesStridedMinClampWithSegmentKernels() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor cpuA = tensor("cpuA");
        Tensor cpuB = tensor("cpuB");
        Tensor cpuBias = bias("cpuBias");
        Tensor cpuOut = cpuA.matmul(cpuB).transpose().min(cpuBias).clampMin(-0.25d).clampMax(0.75d);
        CompiledGraph.compile(cpuOut, noSemanticLinearFusion())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor a = tensor("a");
        Tensor b = tensor("b");
        Tensor bias = bias("bias");
        Tensor out = a.matmul(b).transpose().min(bias).clampMin(-0.25d).clampMax(0.75d);

        var trace = CompiledGraph.compile(out, noSemanticLinearFusion())
                .executeTraced(
                        openBlasRuntime(CpuStorageProfile.CPU_NATIVE),
                        ExecutionMode.FORWARD,
                        PublicationPolicy.NONE
                );
        var step = trace.steps().stream()
                .filter(candidate -> candidate.metadata().attributes().containsKey("nativeCpuRegionId"))
                .findFirst()
                .orElseThrow();

        assertEquals("NATIVE", step.metadata().attributes().get("nativeCpuRegionRoute"));
        assertEquals("", step.metadata().attributes().get("nativeCpuRegionFallbackReason"));
        assertEquals(3, step.metadata().attributes().get("nativeCpuRegionLocalKernelCount"));
        assertEquals(1, step.metadata().attributes().get("nativeCpuRegionLocalViewCount"));
        assertEquals(1L, step.metadata().attributes().get("nativeCpuStridedNodeCount"));
        assertTrue(((List<?>) step.metadata().attributes().get("nativeCpuRegionSegmentKernelFamilies"))
                .contains("SEGMENT_STRIDED_SCALAR"));

        CompiledGraph.compile(out, noSemanticLinearFusion())
                .execute(openBlasRuntime(CpuStorageProfile.CPU_NATIVE), ExecutionMode.FORWARD);
        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 1e-4);
    }

    @Test
    void cpuNativeRegionExecutesStridedPowAndPowTensorWithSegmentKernels() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor cpuA = tensor("cpuA");
        Tensor cpuB = tensor("cpuB");
        Tensor cpuExponent = bias("cpuExponent");
        Tensor cpuOut = cpuA.matmul(cpuB).transpose().abs().pow(2.0d).pow(cpuExponent);
        CompiledGraph.compile(cpuOut, noSemanticLinearFusion())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor a = tensor("a");
        Tensor b = tensor("b");
        Tensor exponent = bias("exponent");
        Tensor out = a.matmul(b).transpose().abs().pow(2.0d).pow(exponent);

        var trace = CompiledGraph.compile(out, noSemanticLinearFusion())
                .executeTraced(
                        openBlasRuntime(CpuStorageProfile.CPU_NATIVE),
                        ExecutionMode.FORWARD,
                        PublicationPolicy.NONE
                );
        var step = trace.steps().stream()
                .filter(candidate -> candidate.metadata().attributes().containsKey("nativeCpuRegionId"))
                .findFirst()
                .orElseThrow();

        assertEquals("NATIVE", step.metadata().attributes().get("nativeCpuRegionRoute"));
        assertEquals("", step.metadata().attributes().get("nativeCpuRegionFallbackReason"));
        assertEquals(3, step.metadata().attributes().get("nativeCpuRegionLocalKernelCount"));
        assertEquals(1, step.metadata().attributes().get("nativeCpuRegionLocalViewCount"));
        assertEquals(1L, step.metadata().attributes().get("nativeCpuStridedNodeCount"));
        assertTrue(((List<?>) step.metadata().attributes().get("nativeCpuRegionSegmentKernelFamilies"))
                .contains("SEGMENT_STRIDED_SCALAR"));

        CompiledGraph.compile(out, noSemanticLinearFusion())
                .execute(openBlasRuntime(CpuStorageProfile.CPU_NATIVE), ExecutionMode.FORWARD);
        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 1e-4);
    }

    @Test
    void cpuNativeRegionExecutesFloorCeilSignWithSegmentKernels() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor cpuA = tensor("cpuA");
        Tensor cpuB = tensor("cpuB");
        Tensor cpuOut = cpuA.matmul(cpuB).transpose().clampMin(10.0d).floor().ceil().sign();
        CompiledGraph.compile(cpuOut, noSemanticLinearFusion())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor a = tensor("a");
        Tensor b = tensor("b");
        Tensor out = a.matmul(b).transpose().clampMin(10.0d).floor().ceil().sign();

        var trace = CompiledGraph.compile(out, noSemanticLinearFusion())
                .executeTraced(
                        openBlasRuntime(CpuStorageProfile.CPU_NATIVE),
                        ExecutionMode.FORWARD,
                        PublicationPolicy.NONE
                );
        var step = trace.steps().stream()
                .filter(candidate -> candidate.metadata().attributes().containsKey("nativeCpuRegionId"))
                .findFirst()
                .orElseThrow();

        assertEquals("NATIVE", step.metadata().attributes().get("nativeCpuRegionRoute"));
        assertEquals("", step.metadata().attributes().get("nativeCpuRegionFallbackReason"));
        assertEquals(4, step.metadata().attributes().get("nativeCpuRegionLocalKernelCount"));
        assertEquals(1, step.metadata().attributes().get("nativeCpuRegionLocalViewCount"));
        assertEquals(1L, step.metadata().attributes().get("nativeCpuStridedNodeCount"));
        assertTrue(((List<?>) step.metadata().attributes().get("nativeCpuRegionSegmentKernelFamilies"))
                .contains("SEGMENT_STRIDED_SCALAR"));

        CompiledGraph.compile(out, noSemanticLinearFusion())
                .execute(openBlasRuntime(CpuStorageProfile.CPU_NATIVE), ExecutionMode.FORWARD);
        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 1e-4);
    }

    @Test
    void cpuNativeRegionExecutesStridedWhereWithSegmentKernel() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor cpuCondition = boolColumnVector("cpuCondition");
        Tensor cpuFallback = tensor("cpuFallback");
        Tensor cpuA = tensor("cpuA");
        Tensor cpuB = tensor("cpuB");
        Tensor cpuOut = Tensor.where(cpuCondition, cpuA.matmul(cpuB).transpose(), cpuFallback);
        CompiledGraph.compile(cpuOut, noSemanticLinearFusion())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor condition = boolColumnVector("condition");
        Tensor fallback = tensor("fallback");
        Tensor a = tensor("a");
        Tensor b = tensor("b");
        Tensor out = Tensor.where(condition, a.matmul(b).transpose(), fallback);

        var trace = CompiledGraph.compile(out, noSemanticLinearFusion())
                .executeTraced(
                        openBlasRuntime(CpuStorageProfile.CPU_NATIVE),
                        ExecutionMode.FORWARD,
                        PublicationPolicy.NONE
                );
        var step = trace.steps().stream()
                .filter(candidate -> candidate.metadata().attributes().containsKey("nativeCpuRegionId"))
                .findFirst()
                .orElseThrow();

        assertEquals("NATIVE", step.metadata().attributes().get("nativeCpuRegionRoute"));
        assertEquals("", step.metadata().attributes().get("nativeCpuRegionFallbackReason"));
        assertEquals(1, step.metadata().attributes().get("nativeCpuRegionLocalKernelCount"));
        assertEquals(1, step.metadata().attributes().get("nativeCpuRegionLocalViewCount"));
        assertEquals(3, step.metadata().attributes().get("nativeCpuRegionExecutedGroupCount"));
        assertEquals(1L, step.metadata().attributes().get("nativeCpuStridedNodeCount"));
        assertTrue(((List<?>) step.metadata().attributes().get("nativeCpuRegionSegmentKernelFamilies"))
                .contains("SEGMENT_STRIDED_SCALAR"));

        CompiledGraph.compile(out, noSemanticLinearFusion())
                .execute(openBlasRuntime(CpuStorageProfile.CPU_NATIVE), ExecutionMode.FORWARD);
        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 1e-4);
    }

    @Test
    void cpuNativeRegionExecutesStridedCompareWithSegmentKernelAndCpuBoolBoundary() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor cpuA = tensor("cpuA");
        Tensor cpuB = tensor("cpuB");
        Tensor cpuThreshold = bias("cpuThreshold");
        Tensor cpuOut = cpuA.matmul(cpuB).transpose().greaterThan(cpuThreshold);
        CompiledGraph.compile(cpuOut, noSemanticLinearFusion())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor a = tensor("a");
        Tensor b = tensor("b");
        Tensor threshold = bias("threshold");
        Tensor out = a.matmul(b).transpose().greaterThan(threshold);

        var trace = CompiledGraph.compile(out, noSemanticLinearFusion())
                .executeTraced(
                        openBlasRuntime(CpuStorageProfile.CPU_NATIVE),
                        ExecutionMode.FORWARD,
                        PublicationPolicy.NONE
                );
        var step = trace.steps().stream()
                .filter(candidate -> candidate.metadata().attributes().containsKey("nativeCpuRegionId"))
                .findFirst()
                .orElseThrow();

        assertEquals("NATIVE", step.metadata().attributes().get("nativeCpuRegionRoute"));
        assertEquals("", step.metadata().attributes().get("nativeCpuRegionFallbackReason"));
        assertEquals(1, step.metadata().attributes().get("nativeCpuRegionLocalKernelCount"));
        assertEquals(1, step.metadata().attributes().get("nativeCpuRegionLocalViewCount"));
        assertEquals(3, step.metadata().attributes().get("nativeCpuRegionExecutedGroupCount"));
        assertEquals(1L, step.metadata().attributes().get("nativeCpuStridedNodeCount"));
        assertTrue(((List<?>) step.metadata().attributes().get("nativeCpuRegionSegmentKernelFamilies"))
                .contains("SEGMENT_STRIDED_SCALAR"));

        CompiledGraph.compile(out, noSemanticLinearFusion())
                .execute(openBlasRuntime(CpuStorageProfile.CPU_NATIVE), ExecutionMode.FORWARD);
        assertArrayEquals(cpuOut.getBoolData(), out.getBoolData());
    }

    @Test
    void cpuNativeRegionKeepsCompareWhereMaskNativeUntilBoundaryPublication() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor cpuA = tensor("cpuA");
        Tensor cpuB = tensor("cpuB");
        Tensor cpuThreshold = bias("cpuThreshold");
        Tensor cpuFallback = tensor("cpuFallback");
        Tensor cpuActivations = cpuA.matmul(cpuB);
        Tensor cpuMask = cpuActivations.greaterThan(cpuThreshold);
        Tensor cpuOut = Tensor.where(cpuMask, cpuActivations, cpuFallback).mean(1, false);
        CompiledGraph.compile(cpuOut, noSemanticLinearFusion())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor a = tensor("a");
        Tensor b = tensor("b");
        Tensor threshold = bias("threshold");
        Tensor fallback = tensor("fallback");
        Tensor activations = a.matmul(b);
        Tensor mask = activations.greaterThan(threshold);
        Tensor out = Tensor.where(mask, activations, fallback).mean(1, false);

        var trace = CompiledGraph.compile(out, noSemanticLinearFusion())
                .executeTraced(
                        openBlasRuntime(CpuStorageProfile.CPU_NATIVE),
                        ExecutionMode.FORWARD,
                        PublicationPolicy.NONE
                );
        var step = trace.steps().stream()
                .filter(candidate -> candidate.metadata().attributes().containsKey("nativeCpuRegionId"))
                .findFirst()
                .orElseThrow();

        assertEquals("NATIVE", step.metadata().attributes().get("nativeCpuRegionRoute"));
        assertEquals("", step.metadata().attributes().get("nativeCpuRegionFallbackReason"));
        assertTrue(((List<?>) step.metadata().attributes().get("nativeCpuParityResultResidencies")).stream()
                .anyMatch(row -> row instanceof List<?> values && values.contains("BOOL_MASK_NATIVE")));
        assertFalse(trace.cpuMaterializations().stream()
                .anyMatch(materialization -> materialization.detail().contains("bool_mask_published")));

        CompiledGraph.compile(out, noSemanticLinearFusion())
                .execute(openBlasRuntime(CpuStorageProfile.CPU_NATIVE), ExecutionMode.FORWARD);
        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 1e-4);
    }

    @Test
    void cpuNativeRegionKeepsLogicalMaskNativeBetweenCompareAndWhere() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor cpuA = tensor("cpuA");
        Tensor cpuB = tensor("cpuB");
        Tensor cpuThreshold = bias("cpuThreshold");
        Tensor cpuFallback = tensor("cpuFallback");
        Tensor cpuActivations = cpuA.matmul(cpuB);
        Tensor cpuMask = cpuActivations.greaterThan(cpuThreshold).logicalNot();
        Tensor cpuOut = Tensor.where(cpuMask, cpuActivations, cpuFallback).mean(1, false);
        CompiledGraph.compile(cpuOut, noSemanticLinearFusion())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor a = tensor("a");
        Tensor b = tensor("b");
        Tensor threshold = bias("threshold");
        Tensor fallback = tensor("fallback");
        Tensor activations = a.matmul(b);
        Tensor mask = activations.greaterThan(threshold).logicalNot();
        Tensor out = Tensor.where(mask, activations, fallback).mean(1, false);

        var trace = CompiledGraph.compile(out, noSemanticLinearFusion())
                .executeTraced(
                        openBlasRuntime(CpuStorageProfile.CPU_NATIVE),
                        ExecutionMode.FORWARD,
                        PublicationPolicy.NONE
                );
        var step = trace.steps().stream()
                .filter(candidate -> candidate.metadata().attributes().containsKey("nativeCpuRegionId"))
                .findFirst()
                .orElseThrow();

        assertEquals("NATIVE", step.metadata().attributes().get("nativeCpuRegionRoute"));
        assertEquals("", step.metadata().attributes().get("nativeCpuRegionFallbackReason"));
        assertTrue(((List<?>) step.metadata().attributes().get("nativeCpuParityResultResidencies")).stream()
                .anyMatch(row -> row instanceof List<?> values && values.contains("BOOL_MASK_NATIVE")));
        assertFalse(trace.cpuMaterializations().stream()
                .anyMatch(materialization -> materialization.detail().contains("bool_mask_published")));

        CompiledGraph.compile(out, noSemanticLinearFusion())
                .execute(openBlasRuntime(CpuStorageProfile.CPU_NATIVE), ExecutionMode.FORWARD);
        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 1e-4);
    }

    @Test
    void cpuNativeRegionExecutesMaskedLossLikeFragmentWithoutMaskPublication() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor cpuA = tensor("cpuA");
        Tensor cpuB = tensor("cpuB");
        Tensor cpuThreshold = bias("cpuThreshold");
        Tensor cpuZero = zeroTensor("cpuZero");
        Tensor cpuActivations = cpuA.matmul(cpuB).relu();
        Tensor cpuMask = cpuActivations.greaterThan(cpuThreshold);
        Tensor cpuOut = Tensor.where(cpuMask, cpuActivations, cpuZero).sum().mul(1.0d / (64.0d * 64.0d));
        CompiledGraph.compile(cpuOut, noSemanticLinearFusion())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor a = tensor("a");
        Tensor b = tensor("b");
        Tensor threshold = bias("threshold");
        Tensor zero = zeroTensor("zero");
        Tensor activations = a.matmul(b).relu();
        Tensor mask = activations.greaterThan(threshold);
        Tensor out = Tensor.where(mask, activations, zero).sum().mul(1.0d / (64.0d * 64.0d));

        var trace = CompiledGraph.compile(out, noSemanticLinearFusion())
                .executeTraced(
                        openBlasRuntime(CpuStorageProfile.CPU_NATIVE),
                        ExecutionMode.FORWARD,
                        PublicationPolicy.NONE
                );
        var step = nativeRegionStep(trace.steps());

        assertEquals("NATIVE", step.metadata().attributes().get("nativeCpuRegionRoute"));
        assertEquals("", step.metadata().attributes().get("nativeCpuRegionFallbackReason"));
        assertEquals(5, step.metadata().attributes().get("nativeCpuRegionLocalKernelCount"));
        assertTrue(((List<?>) step.metadata().attributes().get("nativeCpuParityResultResidencies")).stream()
                .anyMatch(row -> row instanceof List<?> values && values.contains("BOOL_MASK_NATIVE")));
        assertFalse(trace.cpuMaterializations().stream()
                .anyMatch(materialization -> materialization.detail().contains("bool_mask_published")));

        CompiledGraph.compile(out, noSemanticLinearFusion())
                .execute(openBlasRuntime(CpuStorageProfile.CPU_NATIVE), ExecutionMode.FORWARD);
        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 1e-4);
    }

    @Test
    void cpuNativeRegionExecutesBoolMaskReductionAndPublishesBoundaryExplicitly() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor cpuA = tensor("cpuA");
        Tensor cpuB = tensor("cpuB");
        Tensor cpuThreshold = bias("cpuThreshold");
        Tensor cpuOut = cpuA.matmul(cpuB).greaterThan(cpuThreshold).logicalNot().any(1, false);
        CompiledGraph.compile(cpuOut, noSemanticLinearFusion())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor a = tensor("a");
        Tensor b = tensor("b");
        Tensor threshold = bias("threshold");
        Tensor out = a.matmul(b).greaterThan(threshold).logicalNot().any(1, false);

        var trace = CompiledGraph.compile(out, noSemanticLinearFusion())
                .executeTraced(
                        openBlasRuntime(CpuStorageProfile.CPU_NATIVE),
                        ExecutionMode.FORWARD,
                        PublicationPolicy.NONE
                );
        var step = trace.steps().stream()
                .filter(candidate -> candidate.metadata().attributes().containsKey("nativeCpuRegionId"))
                .findFirst()
                .orElseThrow();

        assertEquals("NATIVE", step.metadata().attributes().get("nativeCpuRegionRoute"));
        assertEquals("", step.metadata().attributes().get("nativeCpuRegionFallbackReason"));
        assertTrue(trace.cpuMaterializations().stream()
                .anyMatch(materialization -> materialization.detail().contains("bool_mask_published")));

        CompiledGraph.compile(out, noSemanticLinearFusion())
                .execute(openBlasRuntime(CpuStorageProfile.CPU_NATIVE), ExecutionMode.FORWARD);
        assertArrayEquals(cpuOut.getBoolData(), out.getBoolData());
    }

    @Test
    void cpuNativeRegionExecutesTanhSigmoidAsRegionLocalUnaryKernels() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor cpuA = tensor("cpuA");
        Tensor cpuB = tensor("cpuB");
        Tensor cpuOut = cpuA.matmul(cpuB).tanh().sigmoid();
        CompiledGraph.compile(cpuOut, noSemanticLinearFusion())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor a = tensor("a");
        Tensor b = tensor("b");
        Tensor out = a.matmul(b).tanh().sigmoid();

        var trace = CompiledGraph.compile(out, noSemanticLinearFusion())
                .executeTraced(
                        openBlasRuntime(CpuStorageProfile.CPU_NATIVE),
                        ExecutionMode.FORWARD
                );
        var step = trace.steps().stream()
                .filter(candidate -> candidate.metadata().attributes().containsKey("nativeCpuRegionId"))
                .findFirst()
                .orElseThrow();

        assertEquals("NATIVE", step.metadata().attributes().get("nativeCpuRegionRoute"));
        assertEquals("", step.metadata().attributes().get("nativeCpuRegionFallbackReason"));
        assertEquals(2, step.metadata().attributes().get("nativeCpuRegionLocalKernelCount"));
        assertEquals(2, step.metadata().attributes().get("nativeCpuRegionExecutedGroupCount"));

        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 1e-4);
    }

    @Test
    void cpuNativeRegionExecutesF64BiasAndActivationsAsRegionLocalKernels() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat64GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor cpuA = f64Tensor("cpuA");
        Tensor cpuB = f64Tensor("cpuB");
        Tensor cpuBias = f64Bias("cpuBias");
        Tensor cpuOut = cpuA.matmul(cpuB).add(cpuBias).tanh().sigmoid();
        CompiledGraph.compile(cpuOut, noSemanticLinearFusion())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor a = f64Tensor("a");
        Tensor b = f64Tensor("b");
        Tensor bias = f64Bias("bias");
        Tensor out = a.matmul(b).add(bias).tanh().sigmoid();

        var trace = CompiledGraph.compile(out, noSemanticLinearFusion())
                .executeTraced(
                        openBlasRuntime(CpuStorageProfile.CPU_NATIVE),
                        ExecutionMode.FORWARD
                );
        var step = trace.steps().stream()
                .filter(candidate -> candidate.metadata().attributes().containsKey("nativeCpuRegionId"))
                .findFirst()
                .orElseThrow();

        assertEquals("NATIVE", step.metadata().attributes().get("nativeCpuRegionRoute"));
        assertEquals("", step.metadata().attributes().get("nativeCpuRegionFallbackReason"));
        assertEquals(3, step.metadata().attributes().get("nativeCpuRegionLocalKernelCount"));
        assertEquals(2, step.metadata().attributes().get("nativeCpuRegionExecutedGroupCount"));

        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void cpuNativeRegionExecutesF64WhereAsRegionLocalKernel() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat64GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor cpuCondition = boolTensor("cpuCondition");
        Tensor cpuFallback = f64Tensor("cpuFallback");
        Tensor cpuA = f64Tensor("cpuA");
        Tensor cpuB = f64Tensor("cpuB");
        Tensor cpuOut = Tensor.where(cpuCondition, cpuA.matmul(cpuB), cpuFallback);
        CompiledGraph.compile(cpuOut, noSemanticLinearFusion())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor condition = boolTensor("condition");
        Tensor fallback = f64Tensor("fallback");
        Tensor a = f64Tensor("a");
        Tensor b = f64Tensor("b");
        Tensor out = Tensor.where(condition, a.matmul(b), fallback);

        var trace = CompiledGraph.compile(out, noSemanticLinearFusion())
                .executeTraced(
                        openBlasRuntime(CpuStorageProfile.CPU_NATIVE),
                        ExecutionMode.FORWARD
                );
        var step = trace.steps().stream()
                .filter(candidate -> candidate.metadata().attributes().containsKey("nativeCpuRegionId"))
                .findFirst()
                .orElseThrow();

        assertEquals("NATIVE", step.metadata().attributes().get("nativeCpuRegionRoute"));
        assertEquals("", step.metadata().attributes().get("nativeCpuRegionFallbackReason"));
        assertEquals(1, step.metadata().attributes().get("nativeCpuRegionLocalKernelCount"));
        assertEquals(2, step.metadata().attributes().get("nativeCpuRegionExecutedGroupCount"));

        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void cpuNativeRegionPublishesMultipleBoundaryOutputsForNonNativeConsumer() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor cpuA = tensor("cpuA");
        Tensor cpuB = tensor("cpuB");
        Tensor cpuMatmul = cpuA.matmul(cpuB);
        Tensor cpuOut = cpuMatmul.relu().add(cpuMatmul.erf());
        CompiledGraph.compile(cpuOut, noSemanticLinearFusion())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor a = tensor("a");
        Tensor b = tensor("b");
        Tensor matmul = a.matmul(b);
        Tensor out = matmul.relu().add(matmul.erf());

        var trace = CompiledGraph.compile(out, noSemanticLinearFusion())
                .executeTraced(
                        openBlasRuntime(CpuStorageProfile.CPU_NATIVE),
                        ExecutionMode.FORWARD
                );
        var step = trace.steps().stream()
                .filter(candidate -> candidate.metadata().attributes().containsKey("nativeCpuRegionId"))
                .findFirst()
                .orElseThrow();
        List<?> boundaryOutputs = (List<?>) step.metadata().attributes().get("nativeCpuRegionOutputs");

        assertEquals("NATIVE", step.metadata().attributes().get("nativeCpuRegionRoute"));
        assertEquals("", step.metadata().attributes().get("nativeCpuRegionFallbackReason"));
        assertEquals(2, boundaryOutputs.size());
        assertEquals(1, step.metadata().attributes().get("nativeCpuRegionLocalKernelCount"));
        assertEquals(2, step.metadata().attributes().get("nativeCpuRegionExecutedGroupCount"));
        assertTrue(trace.cpuMaterializations().stream()
                .anyMatch(materialization -> boundaryOutputs.contains(materialization.nodeId())
                        && materialization.reason() == CpuMaterializationReason.CPU_CONSUMER));
        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 1e-4);
    }

    @Test
    void cpuNativeSubregionExecutesAfterUnsupportedPrefixInSameCpuRegion() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor cpuPrefixInput = tensor("cpuPrefixInput");
        Tensor cpuA = tensor("cpuA");
        Tensor cpuB = tensor("cpuB");
        Tensor cpuOut = cpuPrefixInput.erf().add(cpuA.matmul(cpuB).relu().erf());
        CompiledGraph.compile(cpuOut, noSemanticLinearFusion())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor prefixInput = tensor("prefixInput");
        Tensor a = tensor("a");
        Tensor b = tensor("b");
        Tensor out = prefixInput.erf().add(a.matmul(b).relu().erf());

        var trace = CompiledGraph.compile(out, noSemanticLinearFusion())
                .executeTraced(
                        openBlasRuntime(CpuStorageProfile.CPU_NATIVE),
                        ExecutionMode.FORWARD
                );
        var step = trace.steps().stream()
                .filter(candidate -> candidate.metadata().attributes().containsKey("nativeCpuRegionId"))
                .findFirst()
                .orElseThrow();
        List<?> boundaryOutputs = (List<?>) step.metadata().attributes().get("nativeCpuRegionOutputs");

        assertEquals("NATIVE", step.metadata().attributes().get("nativeCpuRegionRoute"));
        assertEquals("", step.metadata().attributes().get("nativeCpuRegionFallbackReason"));
        assertEquals(2, step.metadata().attributes().get("nativeCpuRegionNodeCount"));
        assertEquals(1, ((List<?>) step.metadata().attributes().get("nativeCpuRegionProviderNodes")).size());
        assertEquals(1, ((List<?>) step.metadata().attributes().get("nativeCpuRegionLocalKernelNodes")).size());
        assertEquals(1, boundaryOutputs.size());
        assertTrue(trace.cpuMaterializations().stream()
                .anyMatch(materialization -> boundaryOutputs.contains(materialization.nodeId())
                        && materialization.reason() == CpuMaterializationReason.CPU_CONSUMER));
        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 1e-4);
    }

    @Test
    void cpuNativeRegionExecutesDenseMulAsRegionLocalBinaryKernel() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor cpuA = tensor("cpuA");
        Tensor cpuB = tensor("cpuB");
        Tensor cpuFactor = tensor("cpuFactor");
        Tensor cpuOut = cpuA.matmul(cpuB).mul(cpuFactor);
        CompiledGraph.compile(cpuOut, noSemanticLinearFusion())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor a = tensor("a");
        Tensor b = tensor("b");
        Tensor factor = tensor("factor");
        Tensor out = a.matmul(b).mul(factor);

        var trace = CompiledGraph.compile(out, noSemanticLinearFusion())
                .executeTraced(
                        openBlasRuntime(CpuStorageProfile.CPU_NATIVE),
                        ExecutionMode.FORWARD
                );
        var step = trace.steps().stream()
                .filter(candidate -> candidate.metadata().attributes().containsKey("nativeCpuRegionId"))
                .findFirst()
                .orElseThrow();

        assertEquals("NATIVE", step.metadata().attributes().get("nativeCpuRegionRoute"));
        assertEquals("", step.metadata().attributes().get("nativeCpuRegionFallbackReason"));
        assertEquals(1, step.metadata().attributes().get("nativeCpuRegionLocalKernelCount"));
        assertEquals(2, step.metadata().attributes().get("nativeCpuRegionExecutedGroupCount"));

        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 1e-4);
    }

    @Test
    void cpuNativeRegionExecutesLastDimMulAsRegionLocalBinaryKernel() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor cpuA = tensor("cpuA");
        Tensor cpuB = tensor("cpuB");
        Tensor cpuFactor = bias("cpuFactor");
        Tensor cpuOut = cpuA.matmul(cpuB).mul(cpuFactor);
        CompiledGraph.compile(cpuOut, noSemanticLinearFusion())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor a = tensor("a");
        Tensor b = tensor("b");
        Tensor factor = bias("factor");
        Tensor out = a.matmul(b).mul(factor);

        var trace = CompiledGraph.compile(out, noSemanticLinearFusion())
                .executeTraced(
                        openBlasRuntime(CpuStorageProfile.CPU_NATIVE),
                        ExecutionMode.FORWARD
                );
        var step = trace.steps().stream()
                .filter(candidate -> candidate.metadata().attributes().containsKey("nativeCpuRegionId"))
                .findFirst()
                .orElseThrow();

        assertEquals("NATIVE", step.metadata().attributes().get("nativeCpuRegionRoute"));
        assertEquals("", step.metadata().attributes().get("nativeCpuRegionFallbackReason"));
        assertEquals(1, step.metadata().attributes().get("nativeCpuRegionLocalKernelCount"));
        assertEquals(2, step.metadata().attributes().get("nativeCpuRegionExecutedGroupCount"));

        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 1e-4);
    }

    @Test
    void cpuNativeRegionExecutesSumAsRegionLocalReductionKernel() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor cpuA = tensor("cpuA");
        Tensor cpuB = tensor("cpuB");
        Tensor cpuOut = cpuA.matmul(cpuB).sum();
        CompiledGraph.compile(cpuOut, noSemanticLinearFusion())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor a = tensor("a");
        Tensor b = tensor("b");
        Tensor out = a.matmul(b).sum();

        var trace = CompiledGraph.compile(out, noSemanticLinearFusion())
                .executeTraced(
                        openBlasRuntime(CpuStorageProfile.CPU_NATIVE),
                        ExecutionMode.FORWARD
                );
        var step = trace.steps().stream()
                .filter(candidate -> candidate.metadata().attributes().containsKey("nativeCpuRegionId"))
                .findFirst()
                .orElseThrow();

        assertEquals("NATIVE", step.metadata().attributes().get("nativeCpuRegionRoute"));
        assertEquals("", step.metadata().attributes().get("nativeCpuRegionFallbackReason"));
        assertEquals(1, step.metadata().attributes().get("nativeCpuRegionLocalKernelCount"));
        assertEquals(2, step.metadata().attributes().get("nativeCpuRegionExecutedGroupCount"));

        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 1e-2);
    }

    @Test
    void cpuNativeRegionExecutesAxisMeanAsRegionLocalReductionKernel() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor cpuA = tensor("cpuA");
        Tensor cpuB = tensor("cpuB");
        Tensor cpuOut = cpuA.matmul(cpuB).mean(1);
        CompiledGraph.compile(cpuOut, noSemanticLinearFusion())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor a = tensor("a");
        Tensor b = tensor("b");
        Tensor out = a.matmul(b).mean(1);

        var trace = CompiledGraph.compile(out, noSemanticLinearFusion())
                .executeTraced(
                        openBlasRuntime(CpuStorageProfile.CPU_NATIVE),
                        ExecutionMode.FORWARD
                );
        var step = trace.steps().stream()
                .filter(candidate -> candidate.metadata().attributes().containsKey("nativeCpuRegionId"))
                .findFirst()
                .orElseThrow();

        assertEquals("NATIVE", step.metadata().attributes().get("nativeCpuRegionRoute"));
        assertEquals("", step.metadata().attributes().get("nativeCpuRegionFallbackReason"));
        assertEquals(1, step.metadata().attributes().get("nativeCpuRegionLocalKernelCount"));
        assertEquals(2, step.metadata().attributes().get("nativeCpuRegionExecutedGroupCount"));

        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 1e-4);
    }

    @Test
    void cpuNativeRegionExecutesReduceMinMaxAsRegionLocalReductionKernels() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor cpuA = tensor("cpuA");
        Tensor cpuB = tensor("cpuB");
        Tensor cpuOut = cpuA.matmul(cpuB).min(1, false).max();
        CompiledGraph.compile(cpuOut, noSemanticLinearFusion())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor a = tensor("a");
        Tensor b = tensor("b");
        Tensor out = a.matmul(b).min(1, false).max();

        var trace = CompiledGraph.compile(out, noSemanticLinearFusion())
                .executeTraced(
                        openBlasRuntime(CpuStorageProfile.CPU_NATIVE),
                        ExecutionMode.FORWARD
                );
        var step = trace.steps().stream()
                .filter(candidate -> candidate.metadata().attributes().containsKey("nativeCpuRegionId"))
                .findFirst()
                .orElseThrow();

        assertEquals("NATIVE", step.metadata().attributes().get("nativeCpuRegionRoute"));
        assertEquals("", step.metadata().attributes().get("nativeCpuRegionFallbackReason"));
        assertEquals(2, step.metadata().attributes().get("nativeCpuRegionLocalKernelCount"));
        assertEquals(2, step.metadata().attributes().get("nativeCpuRegionExecutedGroupCount"));

        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 1e-4);
    }

    @Test
    void cpuNativeRegionExecutesWhereAsRegionLocalKernel() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor cpuCondition = boolTensor("cpuCondition");
        Tensor cpuFallback = negativeTensor("cpuFallback");
        Tensor cpuA = tensor("cpuA");
        Tensor cpuB = tensor("cpuB");
        Tensor cpuOut = Tensor.where(cpuCondition, cpuA.matmul(cpuB), cpuFallback);
        CompiledGraph.compile(cpuOut, noSemanticLinearFusion())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor condition = boolTensor("condition");
        Tensor fallback = negativeTensor("fallback");
        Tensor a = tensor("a");
        Tensor b = tensor("b");
        Tensor out = Tensor.where(condition, a.matmul(b), fallback);

        var trace = CompiledGraph.compile(out, noSemanticLinearFusion())
                .executeTraced(
                        openBlasRuntime(CpuStorageProfile.CPU_NATIVE),
                        ExecutionMode.FORWARD
                );
        var step = trace.steps().stream()
                .filter(candidate -> candidate.metadata().attributes().containsKey("nativeCpuRegionId"))
                .findFirst()
                .orElseThrow();

        assertEquals("NATIVE", step.metadata().attributes().get("nativeCpuRegionRoute"));
        assertEquals("", step.metadata().attributes().get("nativeCpuRegionFallbackReason"));
        assertEquals(1, step.metadata().attributes().get("nativeCpuRegionLocalKernelCount"));
        assertEquals(2, step.metadata().attributes().get("nativeCpuRegionExecutedGroupCount"));

        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 1e-4);
    }

    @Test
    void cpuNativeRegionExecutesBroadcastWhereAsRegionLocalKernel() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor cpuCondition = boolColumnVector("cpuCondition");
        Tensor cpuFallback = bias("cpuFallback");
        Tensor cpuA = tensor("cpuA");
        Tensor cpuB = tensor("cpuB");
        Tensor cpuOut = Tensor.where(cpuCondition, cpuA.matmul(cpuB), cpuFallback);
        CompiledGraph.compile(cpuOut, noSemanticLinearFusion())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor condition = boolColumnVector("condition");
        Tensor fallback = bias("fallback");
        Tensor a = tensor("a");
        Tensor b = tensor("b");
        Tensor out = Tensor.where(condition, a.matmul(b), fallback);

        var trace = CompiledGraph.compile(out, noSemanticLinearFusion())
                .executeTraced(
                        openBlasRuntime(CpuStorageProfile.CPU_NATIVE),
                        ExecutionMode.FORWARD
                );
        var step = trace.steps().stream()
                .filter(candidate -> candidate.metadata().attributes().containsKey("nativeCpuRegionId"))
                .findFirst()
                .orElseThrow();

        assertEquals("NATIVE", step.metadata().attributes().get("nativeCpuRegionRoute"));
        assertEquals("", step.metadata().attributes().get("nativeCpuRegionFallbackReason"));
        assertEquals(1, step.metadata().attributes().get("nativeCpuRegionLocalKernelCount"));
        assertEquals(2, step.metadata().attributes().get("nativeCpuRegionExecutedGroupCount"));

        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 1e-4);
    }

    @Test
    void cpuNativeRegionExecutesColumnBroadcastMulAsRegionLocalBinaryKernel() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor cpuA = tensor("cpuA");
        Tensor cpuB = tensor("cpuB");
        Tensor cpuFactor = columnVector("cpuFactor");
        Tensor cpuOut = cpuA.matmul(cpuB).mul(cpuFactor);
        CompiledGraph.compile(cpuOut, noSemanticLinearFusion())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor a = tensor("a");
        Tensor b = tensor("b");
        Tensor factor = columnVector("factor");
        Tensor out = a.matmul(b).mul(factor);

        var trace = CompiledGraph.compile(out, noSemanticLinearFusion())
                .executeTraced(
                        openBlasRuntime(CpuStorageProfile.CPU_NATIVE),
                        ExecutionMode.FORWARD
                );
        var step = trace.steps().stream()
                .filter(candidate -> candidate.metadata().attributes().containsKey("nativeCpuRegionId"))
                .findFirst()
                .orElseThrow();

        assertEquals("NATIVE", step.metadata().attributes().get("nativeCpuRegionRoute"));
        assertEquals("", step.metadata().attributes().get("nativeCpuRegionFallbackReason"));
        assertEquals(1, step.metadata().attributes().get("nativeCpuRegionLocalKernelCount"));
        assertEquals(2, step.metadata().attributes().get("nativeCpuRegionExecutedGroupCount"));

        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 1e-4);
    }

    @Test
    void cpuNativeRegionExecutesCompareBoundaryAsCpuArrayOutput() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor cpuA = tensor("cpuA");
        Tensor cpuB = tensor("cpuB");
        Tensor cpuThreshold = columnVector("cpuThreshold");
        Tensor cpuOut = cpuA.matmul(cpuB).greaterThan(cpuThreshold);
        CompiledGraph.compile(cpuOut, noSemanticLinearFusion())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor a = tensor("a");
        Tensor b = tensor("b");
        Tensor threshold = columnVector("threshold");
        Tensor out = a.matmul(b).greaterThan(threshold);

        var trace = CompiledGraph.compile(out, noSemanticLinearFusion())
                .executeTraced(
                        openBlasRuntime(CpuStorageProfile.CPU_NATIVE),
                        ExecutionMode.FORWARD
                );
        var step = trace.steps().stream()
                .filter(candidate -> candidate.metadata().attributes().containsKey("nativeCpuRegionId"))
                .findFirst()
                .orElseThrow();

        assertEquals("NATIVE", step.metadata().attributes().get("nativeCpuRegionRoute"));
        assertEquals("", step.metadata().attributes().get("nativeCpuRegionFallbackReason"));
        assertEquals(1, step.metadata().attributes().get("nativeCpuRegionLocalKernelCount"));
        assertEquals(2, step.metadata().attributes().get("nativeCpuRegionExecutedGroupCount"));

        assertArrayEquals(cpuOut.getBoolData(), out.getBoolData());
    }

    @Test
    void cpuNativeRegionExecutesContiguousAsRegionLocalCopy() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor cpuA = tensor("cpuA");
        Tensor cpuB = tensor("cpuB");
        Tensor cpuOut = cpuA.matmul(cpuB).contiguous();
        CompiledGraph.compile(cpuOut, noSemanticLinearFusion())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor a = tensor("a");
        Tensor b = tensor("b");
        Tensor out = a.matmul(b).contiguous();

        var trace = CompiledGraph.compile(out, noSemanticLinearFusion())
                .executeTraced(
                        openBlasRuntime(CpuStorageProfile.CPU_NATIVE),
                        ExecutionMode.FORWARD
                );
        var step = trace.steps().stream()
                .filter(candidate -> candidate.metadata().attributes().containsKey("nativeCpuRegionId"))
                .findFirst()
                .orElseThrow();

        assertEquals("NATIVE", step.metadata().attributes().get("nativeCpuRegionRoute"));
        assertEquals("", step.metadata().attributes().get("nativeCpuRegionFallbackReason"));
        assertEquals(1, step.metadata().attributes().get("nativeCpuRegionLocalKernelCount"));
        assertEquals(2, step.metadata().attributes().get("nativeCpuRegionExecutedGroupCount"));

        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 1e-4);
    }

    @Test
    void cpuNativeRegionExecutesCastAsRegionLocalKernel() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor cpuA = tensor("cpuA");
        Tensor cpuB = tensor("cpuB");
        Tensor cpuOut = cpuA.matmul(cpuB).cast(DataType.BFLOAT16);
        CompiledGraph.compile(cpuOut, noSemanticLinearFusion())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor a = tensor("a");
        Tensor b = tensor("b");
        Tensor out = a.matmul(b).cast(DataType.BFLOAT16);

        var trace = CompiledGraph.compile(out, noSemanticLinearFusion())
                .executeTraced(
                        openBlasRuntime(CpuStorageProfile.CPU_NATIVE),
                        ExecutionMode.FORWARD
                );
        var step = trace.steps().stream()
                .filter(candidate -> candidate.metadata().attributes().containsKey("nativeCpuRegionId"))
                .findFirst()
                .orElseThrow();

        assertEquals("NATIVE", step.metadata().attributes().get("nativeCpuRegionRoute"));
        assertEquals("", step.metadata().attributes().get("nativeCpuRegionFallbackReason"));
        assertEquals(1, step.metadata().attributes().get("nativeCpuRegionLocalKernelCount"));
        assertEquals(2, step.metadata().attributes().get("nativeCpuRegionExecutedGroupCount"));

        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 1e-2);
    }

    @Test
    void cpuNativeRegionExecutesBf16UnaryAndBinaryAsRegionLocalKernels() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor cpuA = tensor("cpuA");
        Tensor cpuB = tensor("cpuB");
        Tensor cpuFactor = bf16ColumnVector("cpuFactor");
        Tensor cpuOut = cpuA.matmul(cpuB).cast(DataType.BFLOAT16).mul(cpuFactor).relu().abs();
        CompiledGraph.compile(cpuOut, noSemanticLinearFusion())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor a = tensor("a");
        Tensor b = tensor("b");
        Tensor factor = bf16ColumnVector("factor");
        Tensor out = a.matmul(b).cast(DataType.BFLOAT16).mul(factor).relu().abs();

        var trace = CompiledGraph.compile(out, noSemanticLinearFusion())
                .executeTraced(
                        openBlasRuntime(CpuStorageProfile.CPU_NATIVE),
                        ExecutionMode.FORWARD
                );
        var step = trace.steps().stream()
                .filter(candidate -> candidate.metadata().attributes().containsKey("nativeCpuRegionId"))
                .findFirst()
                .orElseThrow();

        assertEquals("NATIVE", step.metadata().attributes().get("nativeCpuRegionRoute"));
        assertEquals("", step.metadata().attributes().get("nativeCpuRegionFallbackReason"));
        assertEquals(4, step.metadata().attributes().get("nativeCpuRegionLocalKernelCount"));
        assertEquals(2, step.metadata().attributes().get("nativeCpuRegionExecutedGroupCount"));

        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 1e-2);
    }

    @Test
    void cpuNativeRegionExecutesBf16SumMeanWithPromotedPrecisionTrace() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor cpuA = tensor("cpuA");
        Tensor cpuB = tensor("cpuB");
        Tensor cpuFactor = bf16ColumnVector("cpuFactor");
        Tensor cpuOut = cpuA.matmul(cpuB).cast(DataType.BFLOAT16).mul(cpuFactor).mean(1, false).sum();
        CompiledGraph.compile(cpuOut, noSemanticLinearFusion())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor a = tensor("a");
        Tensor b = tensor("b");
        Tensor factor = bf16ColumnVector("factor");
        Tensor out = a.matmul(b).cast(DataType.BFLOAT16).mul(factor).mean(1, false).sum();

        var trace = CompiledGraph.compile(out, noSemanticLinearFusion())
                .executeTraced(
                        openBlasRuntime(CpuStorageProfile.CPU_NATIVE),
                        ExecutionMode.FORWARD,
                        PublicationPolicy.NONE
                );
        var step = trace.steps().stream()
                .filter(candidate -> candidate.metadata().attributes().containsKey("nativeCpuRegionId"))
                .findFirst()
                .orElseThrow();

        assertEquals("NATIVE", step.metadata().attributes().get("nativeCpuRegionRoute"));
        assertEquals("", step.metadata().attributes().get("nativeCpuRegionFallbackReason"));
        assertEquals(4, step.metadata().attributes().get("nativeCpuRegionLocalKernelCount"));
        assertEquals(2, step.metadata().attributes().get("nativeCpuRegionExecutedGroupCount"));
        assertEquals("BF16", step.metadata().attributes().get("nativeCpuRegionBf16StoragePrecision"));
        assertEquals("F32_PROMOTED", step.metadata().attributes().get("nativeCpuRegionBf16ComputePrecision"));
        assertTrue(((List<?>) step.metadata().attributes().get("nativeCpuRegionBf16PromotedSegmentScalarNodes")).size() >= 2);

        CompiledGraph.compile(out, noSemanticLinearFusion())
                .execute(openBlasRuntime(CpuStorageProfile.CPU_NATIVE), ExecutionMode.FORWARD);
        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 1e-2);
    }

    @Test
    void cpuNativeRegionExecutesBf16CompareBoundaryAsCpuArrayOutput() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor cpuA = tensor("cpuA");
        Tensor cpuB = tensor("cpuB");
        Tensor cpuThreshold = bf16ColumnVector("cpuThreshold");
        Tensor cpuOut = cpuA.matmul(cpuB).cast(DataType.BFLOAT16).greaterThan(cpuThreshold);
        CompiledGraph.compile(cpuOut, noSemanticLinearFusion())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor a = tensor("a");
        Tensor b = tensor("b");
        Tensor threshold = bf16ColumnVector("threshold");
        Tensor out = a.matmul(b).cast(DataType.BFLOAT16).greaterThan(threshold);

        var trace = CompiledGraph.compile(out, noSemanticLinearFusion())
                .executeTraced(
                        openBlasRuntime(CpuStorageProfile.CPU_NATIVE),
                        ExecutionMode.FORWARD
                );
        var step = trace.steps().stream()
                .filter(candidate -> candidate.metadata().attributes().containsKey("nativeCpuRegionId"))
                .findFirst()
                .orElseThrow();

        assertEquals("NATIVE", step.metadata().attributes().get("nativeCpuRegionRoute"));
        assertEquals("", step.metadata().attributes().get("nativeCpuRegionFallbackReason"));
        assertEquals(2, step.metadata().attributes().get("nativeCpuRegionLocalKernelCount"));
        assertEquals(2, step.metadata().attributes().get("nativeCpuRegionExecutedGroupCount"));

        assertArrayEquals(cpuOut.getBoolData(), out.getBoolData());
    }

    @Test
    void cpuNativeRequireNativeAllowsFullySupportedBf16CompareRegion() {
        Assumptions.assumeTrue(OpenBlasFfmBridge.isFloat32GemmAvailable(), OpenBlasFfmBridge.unavailableReason());

        Tensor cpuA = tensor("cpuA");
        Tensor cpuB = tensor("cpuB");
        Tensor cpuThreshold = bf16ColumnVector("cpuThreshold");
        Tensor cpuOut = cpuA.matmul(cpuB).cast(DataType.BFLOAT16).greaterThan(cpuThreshold);
        CompiledGraph.compile(cpuOut, noSemanticLinearFusion())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor a = tensor("a");
        Tensor b = tensor("b");
        Tensor threshold = bf16ColumnVector("threshold");
        Tensor out = a.matmul(b).cast(DataType.BFLOAT16).greaterThan(threshold);

        var trace = CompiledGraph.compile(out, noSemanticLinearFusion())
                .executeTraced(
                        openBlasRuntime(CpuStorageProfile.CPU_NATIVE)
                                .withNativeCpuFailurePolicy(config.runtime.NativeCpuFailurePolicy.REQUIRE_NATIVE),
                        ExecutionMode.FORWARD
                );
        var step = trace.steps().stream()
                .filter(candidate -> candidate.metadata().attributes().containsKey("nativeCpuRegionId"))
                .findFirst()
                .orElseThrow();

        assertEquals("NATIVE", step.metadata().attributes().get("nativeCpuRegionRoute"));
        assertEquals("", step.metadata().attributes().get("nativeCpuRegionFallbackReason"));
        assertArrayEquals(cpuOut.getBoolData(), out.getBoolData());
    }

    private static RuntimeConfig openBlasRuntime(CpuStorageProfile profile) {
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
        ).withCpuStorageProfile(profile);
    }

    private static CompileConfig noSemanticLinearFusion() {
        return CompileConfig.inference()
                .withSemanticCanonicalization(SemanticCanonicalizationConfig.disabled());
    }

    private static ExecutionStepTrace nativeRegionStep(List<ExecutionStepTrace> steps) {
        return steps.stream()
                .filter(candidate -> candidate.metadata().attributes().containsKey("nativeCpuRegionId"))
                .findFirst()
                .orElseThrow();
    }

    private static Tensor tensor(String label) {
        float[] values = new float[64 * 64];
        for (int i = 0; i < values.length; i++) {
            values[i] = (float) Math.sin(i * 0.013);
        }
        return new Tensor(values, new int[]{64, 64}, null, label, DataType.FLOAT32);
    }

    private static Tensor bias(String label) {
        float[] values = new float[64];
        for (int i = 0; i < values.length; i++) {
            values[i] = (float) Math.cos(i * 0.017);
        }
        return new Tensor(values, new int[]{64}, null, label, DataType.FLOAT32);
    }

    private static Tensor f64Tensor(String label) {
        double[] values = new double[64 * 64];
        for (int i = 0; i < values.length; i++) {
            values[i] = Math.sin(i * 0.013);
        }
        return new Tensor(values, new int[]{64, 64}, null, label, DataType.FLOAT64);
    }

    private static Tensor f64Bias(String label) {
        double[] values = new double[64];
        for (int i = 0; i < values.length; i++) {
            values[i] = Math.cos(i * 0.017);
        }
        return new Tensor(values, new int[]{64}, null, label, DataType.FLOAT64);
    }

    private static Tensor columnVector(String label) {
        float[] values = new float[64];
        for (int i = 0; i < values.length; i++) {
            values[i] = (float) (1.0 + 0.01 * i);
        }
        return new Tensor(values, new int[]{64, 1}, null, label, DataType.FLOAT32);
    }

    private static Tensor bf16ColumnVector(String label) {
        float[] values = new float[64];
        for (int i = 0; i < values.length; i++) {
            values[i] = (float) (1.0 + 0.01 * i);
        }
        return new Tensor(values, new int[]{64, 1}, null, label, DataType.BFLOAT16);
    }

    private static Tensor negativeTensor(String label) {
        float[] values = new float[64 * 64];
        for (int i = 0; i < values.length; i++) {
            values[i] = -10.0f - (i % 17);
        }
        return new Tensor(values, new int[]{64, 64}, null, label, DataType.FLOAT32);
    }

    private static Tensor zeroTensor(String label) {
        return new Tensor(new float[64 * 64], new int[]{64, 64}, null, label, DataType.FLOAT32);
    }

    private static Tensor boolTensor(String label) {
        byte[] values = new byte[64 * 64];
        for (int i = 0; i < values.length; i++) {
            values[i] = (byte) (i % 3 == 0 ? 1 : 0);
        }
        return new Tensor(values, new int[]{64, 64}, null, label, DataType.BOOL);
    }

    private static Tensor boolColumnVector(String label) {
        byte[] values = new byte[64];
        for (int i = 0; i < values.length; i++) {
            values[i] = (byte) (i % 2 == 0 ? 1 : 0);
        }
        return new Tensor(values, new int[]{64, 1}, null, label, DataType.BOOL);
    }
}
