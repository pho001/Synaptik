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
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
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
