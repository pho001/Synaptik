import backend.ComputeBackend;
import backend.accelerator.exec.PartitionExecutionRole;
import backend.accelerator.lowering.GpuCompoundPatternType;
import backend.accelerator.select.AcceleratorPlanCostModel;
import backend.cuda.lowering.CudaGpuRegionLegalityAdapter;
import backend.metal.exec.PreparedMetalExecutable;
import backend.metal.lowering.MetalPartitionSupport;
import backend.cuda.exec.PreparedCudaExecutable;
import backend.runtime.ExecutionMode;
import config.optimizer.OptimizerConfig;
import config.optimizer.OffloadConfig;
import config.optimizer.PartitionConfig;
import config.runtime.AcceleratorBackendConfig;
import config.runtime.AcceleratorBufferBindingMode;
import config.runtime.AcceleratorBufferConfig;
import config.runtime.AcceleratorConfig;
import config.runtime.RuntimeConfig;
import config.runtime.FusedExecutionPolicy;
import config.runtime.FusedPrimaryBackend;
import graph.CompiledGraph;
import graph.execution.PreparedExecution;
import operations.Operation;
import backend.blas.BlasProvider;
import config.backend.KernelTuningConfig;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;

import java.util.List;

import static org.junit.jupiter.api.Assumptions.assumeTrue;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class PreparedExecutionBuildTest {
    @Test
    void inferenceOnlyGraphBuildsForwardOnlyPreparedExecution() {
        Tensor a = new Tensor(new double[]{1.0, 2.0, 3.0, 4.0}, new int[]{4}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{0.5, 1.5, -2.0, 3.0}, new int[]{4}, null, "b", DataType.FLOAT64);
        Tensor out = a.add(b).mul(a).sigmoid();

        RuntimeConfig runtimeConfig = RuntimeConfig.inferenceDefaults();
        CompiledGraph compiledGraph = CompiledGraph.compile(out, OptimizerConfig.noOptimization());
        PreparedExecution execution = compiledGraph.prepare(runtimeConfig);

        assertNotNull(execution);
        assertEquals(runtimeConfig, execution.runtimeConfig());
        assertFalse(execution.supportsBackward());
        assertFalse(execution.forwardSteps().isEmpty());
        assertTrue(execution.backwardSteps().isEmpty());

        execution.execute(ExecutionMode.FORWARD);
        assertEquals(4, out.toDoubleArrayCopy().length);
    }

    @Test
    void repeatedPrepareOnSameCompiledGraphBuildsIndependentPreparedExecutions() {
        Tensor a = new Tensor(new double[]{1.0, 2.0, 3.0}, new int[]{3}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{4.0, 5.0, 6.0}, new int[]{3}, null, "b", DataType.FLOAT64);
        a.setRequiresGrad(true);
        b.setRequiresGrad(true);

        Tensor out = a.mul(b).add(a);
        CompiledGraph compiled = CompiledGraph.compile(out, OptimizerConfig.noOptimization());

        PreparedExecution first = compiled.prepare(RuntimeConfig.trainingDefaults());
        PreparedExecution second = compiled.prepare(RuntimeConfig.trainingDefaults());

        assertNotSame(first, second);
        assertNotSame(first.forwardSteps(), second.forwardSteps());
        assertEquals(first.forwardSteps().size(), second.forwardSteps().size());
        assertEquals(first.backwardSteps().size(), second.backwardSteps().size());

        first.execute(ExecutionMode.FORWARD_BACKWARD);
        Tensor firstGradient = a.getGradient();
        assertNotNull(firstGradient);
        firstGradient.setDataAt(0, 999.0);

        second.execute(ExecutionMode.FORWARD_BACKWARD);
        assertArrayEquals(new double[]{5.0, 6.0, 7.0}, a.getGradient().toDoubleArrayCopy(), 1e-9);
        assertArrayEquals(new double[]{1.0, 2.0, 3.0}, b.getGradient().toDoubleArrayCopy(), 1e-9);
    }

    @Test
    void preparedExecutionStepViewsAreUnmodifiable() {
        Tensor a = new Tensor(new double[]{1.0, 2.0}, new int[]{2}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{3.0, 4.0}, new int[]{2}, null, "b", DataType.FLOAT64);
        Tensor out = a.add(b).relu();

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .prepare(RuntimeConfig.inferenceDefaults());

        assertThrows(UnsupportedOperationException.class, () -> execution.forwardSteps().clear());
        assertThrows(UnsupportedOperationException.class, () -> execution.backwardSteps().clear());
    }

    @Test
    void acceleratorLoweringArtifactsAreCompletedDuringCompile() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{5f, 6f, 7f, 8f}, new int[]{2, 2}, null, "b", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor out = matmul.relu();

        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        OptimizerConfig partitionOnly = OptimizerConfig.inferenceDefaults()
                .withStageOrder(java.util.List.of(config.optimizer.OptimizerStage.PART));
        CompiledGraph compiled = CompiledGraph.compile(out, partitionOnly);

        assertFalse(compiled.compileArtifacts().backendSelectionCandidates().isEmpty());
        assertNotNull(compiled.compileArtifacts().optimizerState());
        assertFalse(compiled.compileArtifacts().optimizerState().optimizedRegions().isEmpty());
        assertNotNull(compiled.compileArtifacts().optimizerState().memoryPlan());
        assertNotNull(compiled.compileArtifacts().memoryPlan());
    }

    @Test
    void acceleratorOffloadCanSelectMetalForCpuFloat32GraphWithoutBackendIntent() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{5f, 6f, 7f, 8f}, new int[]{2, 2}, null, "b", DataType.FLOAT32);
        Tensor out = a.matmul(b).relu();

        OptimizerConfig optimizerConfig = OptimizerConfig.inferenceDefaults()
                .withOffload(OffloadConfig.acceleratorGreedy());
        CompiledGraph compiled = CompiledGraph.compile(out, optimizerConfig);

        assertTrue(compiled.compileArtifacts().compiledNodes().stream()
                .filter(node -> node.operation() != null)
                .allMatch(node -> node.backend() == ComputeBackend.CPU));
        assertTrue(compiled.compileArtifacts().backendSelectionCandidates().stream()
                .anyMatch(candidate -> candidate.plan() != null
                        && candidate.plan().backend() == ComputeBackend.GPU_METAL
                        && candidate.nodeIds().size() >= 2));

        PreparedExecution execution = compiled.prepare(RuntimeConfig.inferenceDefaults());

        var selectedDecision = execution.prepareTrace().backendSelection().decisions().stream()
                .filter(decision -> decision.selected()
                        && decision.selectedBackend() == ComputeBackend.GPU_METAL)
                .findFirst()
                .orElseThrow();
        assertNotNull(selectedDecision.costSummary());
        assertEquals("PROFILE_DERIVED", selectedDecision.costSummary().preset());
        assertFalse(selectedDecision.costSummary().preset().isBlank());
        assertTrue(selectedDecision.costSummary().estimatedTransferBytes() >= 0L);
        assertTrue(execution.forwardSteps().stream()
                .anyMatch(step -> step.metadata().backend() == ComputeBackend.GPU_METAL
                        && step.metadata().acceleratorExecutable() instanceof PreparedMetalExecutable));
    }

    @Test
    void prepareTraceSelectedAcceleratorDecisionCarriesPlannerEvidence() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{5f, 6f, 7f, 8f}, new int[]{2, 2}, null, "b", DataType.FLOAT32);
        Tensor out = a.matmul(b).relu();

        OptimizerConfig optimizerConfig = OptimizerConfig.inferenceDefaults()
                .withOffload(OffloadConfig.acceleratorGreedy());
        PreparedExecution execution = CompiledGraph.compile(out, optimizerConfig)
                .prepare(RuntimeConfig.inferenceDefaults());

        var decision = execution.prepareTrace().backendSelection().decisions().stream()
                .filter(candidate -> candidate.selected()
                        && candidate.selectedBackend() == ComputeBackend.GPU_METAL)
                .findFirst()
                .orElseThrow();

        assertNotNull(decision.costSummary());
        assertTrue(decision.costSummary().boundaryCount() >= 0);
        assertTrue(decision.costSummary().estimatedTransferBytes() >= 0L);
        assertTrue(decision.costSummary().estimatedComputeWork() >= 0L);
        assertFalse(decision.costSummary().preset().isBlank());
    }

    @Test
    void metalSelectionKeepsLinearLogSoftmaxInGpuRegion() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "metalLinearLogSoftmaxInput", DataType.FLOAT32);
        Tensor weight = new Tensor(new float[]{1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f}, new int[]{3, 3}, null, "metalLinearLogSoftmaxWeight", DataType.FLOAT32);
        Tensor matmul = input.matmul(weight);
        Tensor out = matmul.logSoftmax(1);
        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        CompiledGraph compiled = CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults());
        PreparedExecution execution = compiled.prepare(RuntimeConfig.inferenceDefaults());
        int matmulNodeId = nodeId(compiled, Operation.OpType.MATMUL);
        int logSoftmaxNodeId = nodeId(compiled, Operation.OpType.LOG_SOFTMAX);

        var selected = execution.prepareTrace().backendSelection().decisions().stream()
                .filter(decision -> decision.selected() && decision.selectedBackend() == ComputeBackend.GPU_METAL)
                .filter(decision -> decision.nodeIds().contains(matmulNodeId) && decision.nodeIds().contains(logSoftmaxNodeId))
                .findFirst()
                .orElseThrow();

        assertEquals("selected", selected.reason());
    }

    @Test
    void cudaSelectionKeepsLinearLogSoftmaxInGpuRegion() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "cudaLinearLogSoftmaxInput", DataType.FLOAT32);
        Tensor weight = new Tensor(new float[]{1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f}, new int[]{3, 3}, null, "cudaLinearLogSoftmaxWeight", DataType.FLOAT32);
        Tensor matmul = input.matmul(weight);
        Tensor out = matmul.logSoftmax(1);
        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_CUDA);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_CUDA);

        CompiledGraph compiled = CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults());
        PreparedExecution execution = compiled.prepare(RuntimeConfig.inferenceDefaults());
        int matmulNodeId = nodeId(compiled, Operation.OpType.MATMUL);
        int logSoftmaxNodeId = nodeId(compiled, Operation.OpType.LOG_SOFTMAX);

        var selected = execution.prepareTrace().backendSelection().decisions().stream()
                .filter(decision -> decision.selected() && decision.selectedBackend() == ComputeBackend.GPU_CUDA)
                .filter(decision -> decision.nodeIds().contains(matmulNodeId) && decision.nodeIds().contains(logSoftmaxNodeId))
                .findFirst()
                .orElseThrow();

        assertEquals("selected", selected.reason());
    }

    @Test
    void metalSelectionRejectsUnsupportedLossAdjacentCandidateVisibly() {
        Tensor logits = new Tensor(new float[]{1f, 2f, 3f, 1f, 0f, -1f}, new int[]{2, 3}, null, "metalRejectedLossLogits", DataType.FLOAT32);
        Tensor targetIndices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "metalRejectedLossTargets", DataType.INT32);
        Tensor out = logits.crossEntropyLossFromIndices(targetIndices, 1);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        CompiledGraph compiled = CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults());
        PreparedExecution execution = compiled.prepare(RuntimeConfig.inferenceDefaults());
        int lossNodeId = nodeId(compiled, Operation.OpType.CROSS_ENTROPY_LOSS_INDICES);
        String plannerReason = MetalPartitionSupport.plannerUnsupportedReason(compiledNode(compiled, lossNodeId), null);

        assertFalse(hasSelectedAcceleratorDecisionFor(execution, ComputeBackend.GPU_METAL, lossNodeId));
        assertTrue(plannerReason.contains("UNSUPPORTED_DTYPE"));
        assertTrue(plannerReason.contains("LOSS") || plannerReason.contains("CROSS_ENTROPY_LOSS_INDICES"));
        assertCpuPreparedStepAvailable(execution, lossNodeId);
    }

    @Test
    void cudaSelectionRejectsUnsupportedReductionCandidateVisibly() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "cudaRejectedReductionInput", DataType.FLOAT32);
        Tensor out = input.sum(1);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_CUDA);

        CompiledGraph compiled = CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults());
        PreparedExecution execution = compiled.prepare(RuntimeConfig.inferenceDefaults());
        int reductionNodeId = nodeId(compiled, Operation.OpType.SUM);
        String plannerReason = CudaGpuRegionLegalityAdapter.plannerUnsupportedReason(compiledNode(compiled, reductionNodeId), null);

        assertFalse(hasSelectedAcceleratorDecisionFor(execution, ComputeBackend.GPU_CUDA, reductionNodeId));
        assertTrue(plannerReason.contains("UNSUPPORTED_OPERATION"));
        assertTrue(plannerReason.contains("SUM") || plannerReason.contains("REDUCTION"));
        assertCpuPreparedStepAvailable(execution, reductionNodeId);
    }

    @Test
    void metalRequiredModeExposesPhaseSeventeenReductionRejection() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "metalRequiredPhase17ReductionInput", DataType.FLOAT32);
        Tensor out = input.sum(1);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        CompiledGraph compiled = CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults());
        PreparedExecution execution = compiled.prepare(runtimeWithRequiredAcceleratorBuffer(ComputeBackend.GPU_METAL));
        int sumNodeId = nodeId(compiled, Operation.OpType.SUM);
        String plannerReason = MetalPartitionSupport.plannerUnsupportedReason(compiledNode(compiled, sumNodeId), null);

        assertFalse(hasSelectedAcceleratorDecisionFor(execution, ComputeBackend.GPU_METAL, sumNodeId));
        assertContainsAll(plannerReason,
                "family=REDUCTION",
                "target=layer_norm_small",
                "operation SUM is not supported");
        assertCpuPreparedStepAvailable(execution, sumNodeId);
    }

    @Test
    void cudaRequiredModeExposesPhaseSeventeenNormalizationRejection() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "cudaRequiredPhase17NormInput", DataType.FLOAT32);
        Tensor gamma = new Tensor(new float[]{1f, 1f}, new int[]{2}, null, "cudaRequiredPhase17NormGamma", DataType.FLOAT32);
        Tensor beta = new Tensor(new float[]{0f, 0f}, new int[]{2}, null, "cudaRequiredPhase17NormBeta", DataType.FLOAT32);
        Tensor out = input.layerNorm(gamma, beta, 1.0e-5);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_CUDA);

        CompiledGraph compiled = CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults());
        PreparedExecution execution = compiled.prepare(runtimeWithRequiredAcceleratorBuffer(ComputeBackend.GPU_CUDA));
        int layerNormNodeId = nodeId(compiled, Operation.OpType.LAYER_NORM);
        String plannerReason = CudaGpuRegionLegalityAdapter.plannerUnsupportedReason(compiledNode(compiled, layerNormNodeId), null);

        assertFalse(hasSelectedAcceleratorDecisionFor(execution, ComputeBackend.GPU_CUDA, layerNormNodeId));
        assertContainsAll(plannerReason,
                "family=NORMALIZATION",
                "target=layer_norm_small",
                "operation LAYER_NORM is not supported");
        assertCpuPreparedStepAvailable(execution, layerNormNodeId);
    }

    @Test
    void phaseSeventeenLogSoftmaxResidualFlowMatchesCpuAndStaysSupported() {
        Tensor cpuInput = new Tensor(new float[]{0.25f, -0.5f, 1.25f, 2f, -1f, 0.75f}, new int[]{2, 3}, null, "phase17CpuLogSoftmaxInput", DataType.FLOAT32);
        Tensor cpuWeight = new Tensor(new float[]{
                1f, 0.5f, -0.25f,
                -0.75f, 1.25f, 0.5f,
                0.25f, -0.5f, 1.5f
        }, new int[]{3, 3}, null, "phase17CpuLogSoftmaxWeight", DataType.FLOAT32);
        Tensor cpuOut = cpuInput.matmul(cpuWeight).logSoftmax(1);
        CompiledGraph.compile(cpuOut, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor input = new Tensor(new float[]{0.25f, -0.5f, 1.25f, 2f, -1f, 0.75f}, new int[]{2, 3}, null, "phase17GpuLogSoftmaxInput", DataType.FLOAT32);
        Tensor weight = new Tensor(new float[]{
                1f, 0.5f, -0.25f,
                -0.75f, 1.25f, 0.5f,
                0.25f, -0.5f, 1.5f
        }, new int[]{3, 3}, null, "phase17GpuLogSoftmaxWeight", DataType.FLOAT32);
        Tensor matmul = input.matmul(weight);
        Tensor out = matmul.logSoftmax(1);
        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        CompiledGraph compiled = CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults());
        PreparedExecution execution = compiled.prepare(RuntimeConfig.inferenceDefaults());
        execution.execute(ExecutionMode.FORWARD);
        int logSoftmaxNodeId = nodeId(compiled, Operation.OpType.LOG_SOFTMAX);

        var manifest = execution.prepareTrace().backendSelection().decisions().stream()
                .filter(decision -> decision.selected() && decision.selectedBackend() == ComputeBackend.GPU_METAL)
                .filter(decision -> decision.nodeIds().contains(logSoftmaxNodeId))
                .map(graph.execution.trace.BackendSelectionDecisionTrace::gpuLoweredRegionManifest)
                .filter(candidate -> candidate != null)
                .findFirst()
                .orElseThrow();
        String manifestText = backend.accelerator.lowering.GpuLoweredRegionManifestRenderer.renderCompact(manifest);

        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 1.0e-5);
        assertContainsAll(manifestText, "LOG_SOFTMAX", "SOFTMAX", "LOG");
    }

    @Test
    void phaseSeventeenCrossEntropyIndexFallbackMatchesCpuAndReportsUnsupportedDType() {
        Tensor cpuLogits = new Tensor(new float[]{1.5f, -0.25f, 0.5f, -1f, 2f, 0.25f}, new int[]{2, 3}, null, "phase17CpuLossLogits", DataType.FLOAT32);
        Tensor cpuTargets = new Tensor(new int[]{0, 1}, new int[]{2}, null, "phase17CpuLossTargets", DataType.INT32);
        Tensor cpuOut = cpuLogits.crossEntropyLossFromIndices(cpuTargets, 1);
        CompiledGraph.compile(cpuOut, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor logits = new Tensor(new float[]{1.5f, -0.25f, 0.5f, -1f, 2f, 0.25f}, new int[]{2, 3}, null, "phase17GpuLossLogits", DataType.FLOAT32);
        Tensor targets = new Tensor(new int[]{0, 1}, new int[]{2}, null, "phase17GpuLossTargets", DataType.INT32);
        Tensor out = logits.crossEntropyLossFromIndices(targets, 1);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        CompiledGraph compiled = CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults());
        PreparedExecution execution = compiled.prepare(RuntimeConfig.inferenceDefaults());
        execution.execute(ExecutionMode.FORWARD);
        int lossNodeId = nodeId(compiled, Operation.OpType.CROSS_ENTROPY_LOSS_INDICES);
        String plannerReason = MetalPartitionSupport.plannerUnsupportedReason(compiledNode(compiled, lossNodeId), null);

        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 1.0e-5);
        assertContainsAll(plannerReason,
                "UNSUPPORTED_DTYPE",
                "family=LOSS_ADJACENT",
                "target=transformer_block_hot_path");
        assertCpuPreparedStepAvailable(execution, lossNodeId);
    }

    @Test
    void phaseSeventeenLayerNormFallbackMatchesCpuAndReportsReductionAdjacent() {
        Tensor cpuInput = new Tensor(new float[]{1f, 2f, 4f, 8f}, new int[]{2, 2}, null, "phase17CpuLayerNormInput", DataType.FLOAT32);
        Tensor cpuGamma = new Tensor(new float[]{1.25f, 0.75f}, new int[]{2}, null, "phase17CpuLayerNormGamma", DataType.FLOAT32);
        Tensor cpuBeta = new Tensor(new float[]{0.5f, -0.25f}, new int[]{2}, null, "phase17CpuLayerNormBeta", DataType.FLOAT32);
        Tensor cpuOut = cpuInput.layerNorm(cpuGamma, cpuBeta, 1.0e-5);
        CompiledGraph.compile(cpuOut, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor input = new Tensor(new float[]{1f, 2f, 4f, 8f}, new int[]{2, 2}, null, "phase17GpuLayerNormInput", DataType.FLOAT32);
        Tensor gamma = new Tensor(new float[]{1.25f, 0.75f}, new int[]{2}, null, "phase17GpuLayerNormGamma", DataType.FLOAT32);
        Tensor beta = new Tensor(new float[]{0.5f, -0.25f}, new int[]{2}, null, "phase17GpuLayerNormBeta", DataType.FLOAT32);
        Tensor out = input.layerNorm(gamma, beta, 1.0e-5);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_CUDA);

        CompiledGraph compiled = CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults());
        PreparedExecution execution = compiled.prepare(RuntimeConfig.inferenceDefaults());
        execution.execute(ExecutionMode.FORWARD);
        int layerNormNodeId = nodeId(compiled, Operation.OpType.LAYER_NORM);
        String plannerReason = CudaGpuRegionLegalityAdapter.plannerUnsupportedReason(compiledNode(compiled, layerNormNodeId), null);

        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 1.0e-5);
        assertContainsAll(plannerReason,
                "REDUCTION_ADJACENT",
                "family=NORMALIZATION",
                "target=layer_norm_small");
        assertCpuPreparedStepAvailable(execution, layerNormNodeId);
    }

    @Test
    void gpuMetalLinearBiasReluCompilesAsOneCompoundRegion() {
        Tensor cpuInput = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "cpuMetalLinearInput", DataType.FLOAT32);
        Tensor cpuWeight = new Tensor(new float[]{
                1f, 0f, 0f, 1f,
                0f, 1f, 1f, 0f,
                1f, 1f, 0f, 0f
        }, new int[]{3, 4}, null, "cpuMetalLinearWeight", DataType.FLOAT32);
        Tensor cpuBias = new Tensor(new float[]{0.5f, -0.5f, 1f, -1f}, new int[]{4}, null, "cpuMetalLinearBias", DataType.FLOAT32);
        Tensor cpuOut = cpuInput.linear(cpuWeight, cpuBias).relu();
        CompiledGraph.compile(cpuOut, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "metalCompoundInput", DataType.FLOAT32);
        Tensor weight = new Tensor(new float[]{
                1f, 0f, 0f, 1f,
                0f, 1f, 1f, 0f,
                1f, 1f, 0f, 0f
        }, new int[]{3, 4}, null, "metalCompoundWeight", DataType.FLOAT32);
        Tensor bias = new Tensor(new float[]{0.5f, -0.5f, 1f, -1f}, new int[]{4}, null, "metalCompoundBias", DataType.FLOAT32);
        Tensor linear = input.linear(weight, bias);
        Tensor out = linear.relu();
        TensorInternalAccess.setBackend(linear, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        CompiledGraph compiled = CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults());
        PreparedExecution execution = compiled.prepare(RuntimeConfig.inferenceDefaults());
        int linearNodeId = nodeId(compiled, Operation.OpType.LINEAR);
        int reluNodeId = nodeId(compiled, Operation.OpType.RELU);
        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .toList();

        assertEquals(1, gpuSteps.size());
        PreparedMetalExecutable executable = (PreparedMetalExecutable) gpuSteps.getFirst().metadata().acceleratorExecutable();
        assertEquals(GpuCompoundPatternType.LINEAR_BIAS_ACTIVATION, executable.compoundSummary().patternType());
        assertTrue(executable.compoundSummary().supported());
        assertTrue(executable.plan().nodeIds().containsAll(List.of(linearNodeId, reluNodeId)));
        assertTrue(executable.plan().lowering().dagSpec().nodes().stream()
                .anyMatch(node -> node.type() == backend.accelerator.dag.AcceleratorDagNodeType.LINEAR));
        assertTrue(executable.plan().lowering().dagSpec().nodes().stream()
                .anyMatch(node -> node.type() == backend.accelerator.dag.AcceleratorDagNodeType.RELU));

        execution.execute(ExecutionMode.FORWARD);

        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 1e-5);
    }

    @Test
    void gpuCudaLinearBiasReluCompilesAsOneCompoundRegion() {
        Tensor cpuInput = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "cpuCudaLinearInput", DataType.FLOAT32);
        Tensor cpuWeight = new Tensor(new float[]{
                1f, 0f, 0f, 1f,
                0f, 1f, 1f, 0f,
                1f, 1f, 0f, 0f
        }, new int[]{3, 4}, null, "cpuCudaLinearWeight", DataType.FLOAT32);
        Tensor cpuBias = new Tensor(new float[]{0.5f, -0.5f, 1f, -1f}, new int[]{4}, null, "cpuCudaLinearBias", DataType.FLOAT32);
        Tensor cpuOut = cpuInput.linear(cpuWeight, cpuBias).relu();
        CompiledGraph.compile(cpuOut, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "cudaCompoundInput", DataType.FLOAT32);
        Tensor weight = new Tensor(new float[]{
                1f, 0f, 0f, 1f,
                0f, 1f, 1f, 0f,
                1f, 1f, 0f, 0f
        }, new int[]{3, 4}, null, "cudaCompoundWeight", DataType.FLOAT32);
        Tensor bias = new Tensor(new float[]{0.5f, -0.5f, 1f, -1f}, new int[]{4}, null, "cudaCompoundBias", DataType.FLOAT32);
        Tensor linear = input.linear(weight, bias);
        Tensor out = linear.relu();
        TensorInternalAccess.setBackend(linear, ComputeBackend.GPU_CUDA);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_CUDA);

        CompiledGraph compiled = CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults());
        PreparedExecution execution = compiled.prepare(RuntimeConfig.inferenceDefaults());
        int linearNodeId = nodeId(compiled, Operation.OpType.LINEAR);
        int reluNodeId = nodeId(compiled, Operation.OpType.RELU);
        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_CUDA)
                .toList();

        assertEquals(1, gpuSteps.size());
        PreparedCudaExecutable executable = (PreparedCudaExecutable) gpuSteps.getFirst().metadata().acceleratorExecutable();
        assertEquals(GpuCompoundPatternType.LINEAR_BIAS_ACTIVATION, executable.compoundSummary().patternType());
        assertTrue(executable.compoundSummary().supported());
        assertTrue(executable.compoundSummary().orderedNodeIds().containsAll(List.of(linearNodeId, reluNodeId)));
        assertTrue(executable.dagSpec().nodes().stream()
                .anyMatch(node -> node.type() == backend.accelerator.dag.AcceleratorDagNodeType.LINEAR));
        assertTrue(executable.dagSpec().nodes().stream()
                .anyMatch(node -> node.type() == backend.accelerator.dag.AcceleratorDagNodeType.RELU));

        execution.execute(ExecutionMode.FORWARD);

        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 1e-5);
    }

    @Test
    void gpuLinearBiasReluRequiredBufferModeFailsBeforeHiddenCpuFallbackWhenUnavailable() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "requiredMetalInput", DataType.FLOAT32);
        Tensor weight = new Tensor(new float[]{
                1f, 0f, 0f, 1f,
                0f, 1f, 1f, 0f,
                1f, 1f, 0f, 0f
        }, new int[]{3, 4}, null, "requiredMetalWeight", DataType.FLOAT32);
        Tensor bias = new Tensor(new float[]{0.5f, -0.5f, 1f, -1f}, new int[]{4}, null, "requiredMetalBias", DataType.FLOAT32);
        Tensor linear = input.linear(weight, bias);
        Tensor out = linear.relu();
        TensorInternalAccess.setBackend(linear, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);
        RuntimeConfig defaults = RuntimeConfig.inferenceDefaults();
        RuntimeConfig runtime = defaults.withAccelerator(defaults.accelerator().withMetal(
                defaults.accelerator().metal().withBuffer(
                        new AcceleratorBufferConfig(AcceleratorBufferBindingMode.REQUIRE, true, Long.MAX_VALUE)
                )
        ));
        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults())
                .prepare(runtime);

        assertEquals(1, execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .count());
        IllegalStateException failure = assertThrows(IllegalStateException.class, () -> execution.execute(ExecutionMode.FORWARD));

        assertTrue(failure.getMessage().contains("Accelerator buffer path is required"));
    }

    @Test
    void metalMatmulBiasActivationEpilogueMatchesCpuAndStaysDeviceOwned() {
        Tensor cpuInput = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "cpuMetalEpilogueInput", DataType.FLOAT32);
        Tensor cpuWeight = new Tensor(new float[]{
                1f, 0f, 0f, 1f,
                0f, 1f, 1f, 0f,
                1f, 1f, 0f, 0f
        }, new int[]{3, 4}, null, "cpuMetalEpilogueWeight", DataType.FLOAT32);
        Tensor cpuBias = new Tensor(new float[]{0.5f, -0.5f, 1f, -1f}, new int[]{4}, null, "cpuMetalEpilogueBias", DataType.FLOAT32);
        Tensor cpuOut = cpuInput.linear(cpuWeight, cpuBias).relu();
        CompiledGraph.compile(cpuOut, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "metalEpilogueInput", DataType.FLOAT32);
        Tensor weight = new Tensor(new float[]{
                1f, 0f, 0f, 1f,
                0f, 1f, 1f, 0f,
                1f, 1f, 0f, 0f
        }, new int[]{3, 4}, null, "metalEpilogueWeight", DataType.FLOAT32);
        Tensor bias = new Tensor(new float[]{0.5f, -0.5f, 1f, -1f}, new int[]{4}, null, "metalEpilogueBias", DataType.FLOAT32);
        Tensor linear = input.linear(weight, bias);
        Tensor out = linear.relu();
        TensorInternalAccess.setBackend(linear, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        CompiledGraph compiled = CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults());
        PreparedExecution execution = compiled.prepare(RuntimeConfig.inferenceDefaults());
        int linearNodeId = nodeId(compiled, Operation.OpType.LINEAR);
        int reluNodeId = nodeId(compiled, Operation.OpType.RELU);
        PreparedMetalExecutable executable = (PreparedMetalExecutable) execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .map(step -> step.metadata().acceleratorExecutable())
                .findFirst()
                .orElseThrow();

        assertTrue(executable.gpuLoweredRegionManifest().fusedSubpatterns().stream()
                .anyMatch(subpattern -> subpattern.patternType() == GpuCompoundPatternType.LINEAR_BIAS_ACTIVATION
                        && subpattern.detail().contains("epilogue")));
        var trace = execution.executeTraced(ExecutionMode.FORWARD);

        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 1.0e-5);
        assertFalse(trace.cpuMaterializations().stream()
                .anyMatch(entry -> (entry.nodeId() == linearNodeId || entry.nodeId() == reluNodeId)
                        && entry.reason() == backend.memory.CpuMaterializationReason.CPU_CONSUMER));
    }

    @Test
    void cudaMatmulBiasActivationEpilogueMatchesCpuAndStaysDeviceOwned() {
        Tensor cpuInput = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "cpuCudaEpilogueInput", DataType.FLOAT32);
        Tensor cpuWeight = new Tensor(new float[]{
                1f, 0f, 0f, 1f,
                0f, 1f, 1f, 0f,
                1f, 1f, 0f, 0f
        }, new int[]{3, 4}, null, "cpuCudaEpilogueWeight", DataType.FLOAT32);
        Tensor cpuBias = new Tensor(new float[]{0.5f, -0.5f, 1f, -1f}, new int[]{4}, null, "cpuCudaEpilogueBias", DataType.FLOAT32);
        Tensor cpuOut = cpuInput.linear(cpuWeight, cpuBias).relu();
        CompiledGraph.compile(cpuOut, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "cudaEpilogueInput", DataType.FLOAT32);
        Tensor weight = new Tensor(new float[]{
                1f, 0f, 0f, 1f,
                0f, 1f, 1f, 0f,
                1f, 1f, 0f, 0f
        }, new int[]{3, 4}, null, "cudaEpilogueWeight", DataType.FLOAT32);
        Tensor bias = new Tensor(new float[]{0.5f, -0.5f, 1f, -1f}, new int[]{4}, null, "cudaEpilogueBias", DataType.FLOAT32);
        Tensor linear = input.linear(weight, bias);
        Tensor out = linear.relu();
        TensorInternalAccess.setBackend(linear, ComputeBackend.GPU_CUDA);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_CUDA);

        CompiledGraph compiled = CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults());
        PreparedExecution execution = compiled.prepare(RuntimeConfig.inferenceDefaults());
        int linearNodeId = nodeId(compiled, Operation.OpType.LINEAR);
        int reluNodeId = nodeId(compiled, Operation.OpType.RELU);
        PreparedCudaExecutable executable = (PreparedCudaExecutable) execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_CUDA)
                .map(step -> step.metadata().acceleratorExecutable())
                .findFirst()
                .orElseThrow();

        assertTrue(executable.gpuLoweredRegionManifest().fusedSubpatterns().stream()
                .anyMatch(subpattern -> subpattern.patternType() == GpuCompoundPatternType.LINEAR_BIAS_ACTIVATION
                        && subpattern.detail().contains("epilogue")));
        var trace = execution.executeTraced(ExecutionMode.FORWARD);

        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 1.0e-5);
        assertFalse(trace.cpuMaterializations().stream()
                .anyMatch(entry -> (entry.nodeId() == linearNodeId || entry.nodeId() == reluNodeId)
                        && entry.reason() == backend.memory.CpuMaterializationReason.CPU_CONSUMER));
    }

    @Test
    void requiredModeReportsEpilogueFusionFallbackInsteadOfSilentCpuReplay() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "requiredEpilogueInput", DataType.FLOAT32);
        Tensor weight = new Tensor(new float[]{
                1f, 0f, 0f, 1f,
                0f, 1f, 1f, 0f,
                1f, 1f, 0f, 0f
        }, new int[]{3, 4}, null, "requiredEpilogueWeight", DataType.FLOAT32);
        Tensor bias = new Tensor(new float[]{0.5f, -0.5f, 1f, -1f}, new int[]{4}, null, "requiredEpilogueBias", DataType.FLOAT32);
        Tensor linear = input.linear(weight, bias);
        Tensor out = linear.relu();
        TensorInternalAccess.setBackend(linear, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);
        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults())
                .prepare(runtimeWithRequiredAcceleratorBuffer(ComputeBackend.GPU_METAL));

        IllegalStateException failure = assertThrows(IllegalStateException.class, () -> execution.execute(ExecutionMode.FORWARD));

        assertTrue(failure.getMessage().contains("Accelerator buffer path is required"));
        assertTrue(failure.getMessage().contains("GPU_METAL"));
    }

    @Test
    void gpuMetalElementwiseChainPublishesCompoundSummary() {
        Tensor a = new Tensor(new float[]{1f, -2f, 3f, -4f}, new int[]{4}, null, "metalChainA", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{0.5f, 1f, -1f, 2f}, new int[]{4}, null, "metalChainB", DataType.FLOAT32);
        Tensor add = a.add(b);
        Tensor relu = add.relu();
        Tensor out = relu.exp();
        TensorInternalAccess.setBackend(add, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(relu, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        CompiledGraph compiled = CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults());
        PreparedExecution execution = compiled.prepare(RuntimeConfig.inferenceDefaults());
        int addNodeId = nodeId(compiled, Operation.OpType.ADD);
        int reluNodeId = nodeId(compiled, Operation.OpType.RELU);
        int expNodeId = nodeId(compiled, Operation.OpType.EXP);
        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .toList();

        assertEquals(1, gpuSteps.size());
        PreparedMetalExecutable executable = (PreparedMetalExecutable) gpuSteps.getFirst().metadata().acceleratorExecutable();
        assertEquals(GpuCompoundPatternType.ELEMENTWISE_CHAIN, executable.compoundSummary().patternType());
        assertTrue(executable.compoundSummary().supported());
        assertTrue(executable.compoundSummary().orderedNodeIds().containsAll(List.of(addNodeId, reluNodeId, expNodeId)));
        assertTrue(executable.compoundSummary().dagNodeTypes().containsAll(List.of("ADD", "RELU", "EXP")));
    }

    @Test
    void gpuCudaElementwiseChainPublishesCompoundSummary() {
        Tensor a = new Tensor(new float[]{1f, -2f, 3f, -4f}, new int[]{4}, null, "cudaChainA", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{0.5f, 1f, -1f, 2f}, new int[]{4}, null, "cudaChainB", DataType.FLOAT32);
        Tensor add = a.add(b);
        Tensor relu = add.relu();
        Tensor out = relu.exp();
        TensorInternalAccess.setBackend(add, ComputeBackend.GPU_CUDA);
        TensorInternalAccess.setBackend(relu, ComputeBackend.GPU_CUDA);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_CUDA);

        CompiledGraph compiled = CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults());
        PreparedExecution execution = compiled.prepare(RuntimeConfig.inferenceDefaults());
        int addNodeId = nodeId(compiled, Operation.OpType.ADD);
        int reluNodeId = nodeId(compiled, Operation.OpType.RELU);
        int expNodeId = nodeId(compiled, Operation.OpType.EXP);
        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_CUDA)
                .toList();

        assertEquals(1, gpuSteps.size());
        PreparedCudaExecutable executable = (PreparedCudaExecutable) gpuSteps.getFirst().metadata().acceleratorExecutable();
        assertEquals(GpuCompoundPatternType.ELEMENTWISE_CHAIN, executable.compoundSummary().patternType());
        assertTrue(executable.compoundSummary().supported());
        assertTrue(executable.compoundSummary().orderedNodeIds().containsAll(List.of(addNodeId, reluNodeId, expNodeId)));
        assertTrue(executable.compoundSummary().dagNodeTypes().containsAll(List.of("ADD", "RELU", "EXP")));
    }

    @Test
    void metalElementwiseFusionKeepsInteriorValuesDeviceOwned() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "metalInteriorA", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 0f, 0f, 1f, 1f, 1f}, new int[]{3, 2}, null, "metalInteriorB", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor relu = matmul.relu();
        Tensor exp = relu.exp();
        Tensor out = exp.log();
        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(relu, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(exp, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        CompiledGraph compiled = CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults());
        PreparedExecution execution = compiled.prepare(RuntimeConfig.inferenceDefaults());
        int reluNodeId = nodeId(compiled, Operation.OpType.RELU);
        int expNodeId = nodeId(compiled, Operation.OpType.EXP);
        int logNodeId = nodeId(compiled, Operation.OpType.LOG);
        PreparedMetalExecutable executable = (PreparedMetalExecutable) execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .map(step -> step.metadata().acceleratorExecutable())
                .findFirst()
                .orElseThrow();

        assertNotNull(executable);
        assertTrue(executable.gpuLoweredRegionManifest().fusedSubpatterns().stream()
                .anyMatch(subpattern -> subpattern.patternType() == GpuCompoundPatternType.ELEMENTWISE_CHAIN
                        && subpattern.originalOperationNodeIds().equals(List.of(reluNodeId, expNodeId, logNodeId))
                        && subpattern.loweredPrimitiveCount() == 3));
        var trace = execution.executeTraced(ExecutionMode.FORWARD);

        assertFalse(trace.cpuMaterializations().stream()
                .anyMatch(entry -> (entry.nodeId() == reluNodeId || entry.nodeId() == expNodeId)
                        && entry.reason() == backend.memory.CpuMaterializationReason.CPU_CONSUMER));
        PreparedExecution required = compiled.prepare(runtimeWithRequiredAcceleratorBuffer(ComputeBackend.GPU_METAL));
        assertTrue(required.forwardSteps().stream()
                .anyMatch(step -> step.metadata().backend() == ComputeBackend.GPU_METAL
                        && step.metadata().acceleratorExecutable() != null));
    }

    @Test
    void cudaElementwiseFusionKeepsInteriorValuesDeviceOwned() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "cudaInteriorA", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 0f, 0f, 1f, 1f, 1f}, new int[]{3, 2}, null, "cudaInteriorB", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor relu = matmul.relu();
        Tensor exp = relu.exp();
        Tensor out = exp.log();
        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_CUDA);
        TensorInternalAccess.setBackend(relu, ComputeBackend.GPU_CUDA);
        TensorInternalAccess.setBackend(exp, ComputeBackend.GPU_CUDA);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_CUDA);

        CompiledGraph compiled = CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults());
        PreparedExecution execution = compiled.prepare(RuntimeConfig.inferenceDefaults());
        int reluNodeId = nodeId(compiled, Operation.OpType.RELU);
        int expNodeId = nodeId(compiled, Operation.OpType.EXP);
        int logNodeId = nodeId(compiled, Operation.OpType.LOG);
        PreparedCudaExecutable executable = (PreparedCudaExecutable) execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_CUDA)
                .map(step -> step.metadata().acceleratorExecutable())
                .findFirst()
                .orElseThrow();

        assertNotNull(executable);
        assertTrue(executable.gpuLoweredRegionManifest().fusedSubpatterns().stream()
                .anyMatch(subpattern -> subpattern.patternType() == GpuCompoundPatternType.ELEMENTWISE_CHAIN
                        && subpattern.originalOperationNodeIds().equals(List.of(reluNodeId, expNodeId, logNodeId))
                        && subpattern.loweredPrimitiveCount() == 3));
        var trace = execution.executeTraced(ExecutionMode.FORWARD);

        assertFalse(trace.cpuMaterializations().stream()
                .anyMatch(entry -> (entry.nodeId() == reluNodeId || entry.nodeId() == expNodeId)
                        && entry.reason() == backend.memory.CpuMaterializationReason.CPU_CONSUMER));
        PreparedExecution required = compiled.prepare(runtimeWithRequiredAcceleratorBuffer(ComputeBackend.GPU_CUDA));
        assertTrue(required.forwardSteps().stream()
                .anyMatch(step -> step.metadata().backend() == ComputeBackend.GPU_CUDA
                        && step.metadata().acceleratorExecutable() != null));
    }

    @Test
    void scoredAcceleratorPlanningRecordsCostSummaryAndBoundedFinalists() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{5f, 6f, 7f, 8f}, new int[]{2, 2}, null, "b", DataType.FLOAT32);
        Tensor out = a.matmul(b).relu();

        OptimizerConfig optimizerConfig = OptimizerConfig.inferenceDefaults()
                .withOffload(OffloadConfig.acceleratorScored());
        CompiledGraph compiled = CompiledGraph.compile(out, optimizerConfig);

        var scoredDecision = compiled.compileTrace().partitionPlanning().decisions().stream()
                .filter(decision -> decision.costSummary() != null)
                .findFirst()
                .orElseThrow();

        assertFalse(scoredDecision.costSummary().preset().isBlank());
        assertTrue(scoredDecision.costSummary().estimatedTransferBytes() >= 0L);
        assertTrue(scoredDecision.costSummary().estimatedComputeWork() >= 0L);
        assertTrue(scoredDecision.finalists().size() <= 3);
    }

    @Test
    void bfloat16FusedPrepareSkipsCompiledAsmKernel() {
        Tensor a = new Tensor(new double[]{1.0, 2.0, 3.0, 4.0}, new int[]{4}, null, "a", DataType.BFLOAT16);
        Tensor b = new Tensor(new double[]{0.5, 1.5, -2.0, 3.0}, new int[]{4}, null, "b", DataType.BFLOAT16);
        Tensor out = a.add(b).mul(a).sigmoid();

        PreparedExecution execution = CompiledGraph.compile(out, fuseOnlyInferenceConfig())
                .prepare(RuntimeConfig.inferenceDefaults());

        var fusedStep = execution.forwardSteps().stream()
                .filter(step -> step.metadata().fusedExecutable() != null)
                .findFirst()
                .orElseThrow();

        assertEquals("BFLOAT16", fusedStep.metadata().cpuPlan().computeContract().storageType().name());
        assertEquals("F32", fusedStep.metadata().cpuPlan().computeContract().computeType().name());
        assertEquals("CPU_FUSED", fusedStep.metadata().cpuPlan().computeContract().backend().name());
        assertNotNull(fusedStep.metadata().fusedExecutable());
    }

    @Test
    void float32FusedPrepareUsesAsmExecutable() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{4}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{0.5f, 1.5f, -2f, 3f}, new int[]{4}, null, "b", DataType.FLOAT32);
        Tensor out = a.add(b).mul(a).sigmoid();

        PreparedExecution execution = CompiledGraph.compile(out, fuseOnlyInferenceConfig())
                .prepare(new RuntimeConfig(
                        kernelWithVectorMin(1),
                        config.runtime.ApproximationConfig.defaults(),
                        config.runtime.BlasConfig.disabled(),
                        new FusedExecutionPolicy(FusedPrimaryBackend.ASM, true)
                ));

        var fusedStep = execution.forwardSteps().stream()
                .filter(step -> step.metadata().fusedExecutable() != null)
                .findFirst()
                .orElseThrow();

        assertEquals("FLOAT32", fusedStep.metadata().cpuPlan().computeContract().storageType().name());
        assertEquals("F32", fusedStep.metadata().cpuPlan().computeContract().computeType().name());
        assertEquals("CPU_FUSED", fusedStep.metadata().cpuPlan().computeContract().backend().name());
        assertTrue(fusedStep.metadata().fusedExecutable().getClass().getName().startsWith("backend.cpu.fused.asm."));
    }

    @Test
    void float64FusedPrepareUsesAsmExecutable() {
        Tensor a = new Tensor(new double[]{1.0, 2.0, 3.0, 4.0}, new int[]{4}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{0.5, 1.5, -2.0, 3.0}, new int[]{4}, null, "b", DataType.FLOAT64);
        Tensor out = a.add(b).mul(a).sigmoid();

        PreparedExecution execution = CompiledGraph.compile(out, fuseOnlyInferenceConfig())
                .prepare(new RuntimeConfig(
                        kernelWithVectorMin(1),
                        config.runtime.ApproximationConfig.defaults(),
                        config.runtime.BlasConfig.disabled(),
                        new FusedExecutionPolicy(FusedPrimaryBackend.ASM, true)
                ));

        var fusedStep = execution.forwardSteps().stream()
                .filter(step -> step.metadata().fusedExecutable() != null)
                .findFirst()
                .orElseThrow();

        assertEquals("FLOAT64", fusedStep.metadata().cpuPlan().computeContract().storageType().name());
        assertEquals("F64", fusedStep.metadata().cpuPlan().computeContract().computeType().name());
        assertEquals("CPU_FUSED", fusedStep.metadata().cpuPlan().computeContract().backend().name());
        assertTrue(fusedStep.metadata().fusedExecutable().getClass().getName().startsWith("backend.cpu.fused.asm."));
    }

    @Test
    void fusedAsmExecutableCacheSeparatesWidthSpecializations() {
        Tensor a1 = new Tensor(new double[]{1.0, 2.0, 3.0, 4.0}, new int[]{4}, null, "a1", DataType.FLOAT64);
        Tensor b1 = new Tensor(new double[]{0.5, 1.5, -2.0, 3.0}, new int[]{4}, null, "b1", DataType.FLOAT64);
        Tensor out1 = a1.add(b1).mul(a1).sigmoid();

        PreparedExecution width1Execution = CompiledGraph.compile(out1, fuseOnlyInferenceConfig())
                .prepare(runtimeWithFusedAsmWidth(1));

        var width1Fused = width1Execution.forwardSteps().stream()
                .filter(step -> step.metadata().fusedExecutable() != null)
                .findFirst()
                .orElseThrow();

        Tensor a2 = new Tensor(new double[]{1.0, 2.0, 3.0, 4.0}, new int[]{4}, null, "a2", DataType.FLOAT64);
        Tensor b2 = new Tensor(new double[]{0.5, 1.5, -2.0, 3.0}, new int[]{4}, null, "b2", DataType.FLOAT64);
        Tensor out2 = a2.add(b2).mul(a2).sigmoid();

        PreparedExecution width2Execution = CompiledGraph.compile(out2, fuseOnlyInferenceConfig())
                .prepare(runtimeWithFusedAsmWidth(2));

        var width2Fused = width2Execution.forwardSteps().stream()
                .filter(step -> step.metadata().fusedExecutable() != null)
                .findFirst()
                .orElseThrow();

        assertEquals(1, width1Fused.metadata().cpuPlan().dispatchHints().vectorWidth());
        assertEquals(2, width2Fused.metadata().cpuPlan().dispatchHints().vectorWidth());
        assertNotEquals(
                width1Fused.metadata().fusedExecutable().getClass().getName(),
                width2Fused.metadata().fusedExecutable().getClass().getName()
        );
        assertTrue(width1Fused.metadata().fusedExecutable().getClass().getName().endsWith("W1"));
        assertTrue(width2Fused.metadata().fusedExecutable().getClass().getName().endsWith("W2"));
    }

    @Test
    void float32FusedPrepareCanBeForcedToAsmFallbackByPolicy() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{4}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{0.5f, 1.5f, -2f, 3f}, new int[]{4}, null, "b", DataType.FLOAT32);
        Tensor out = a.add(b).mul(a).sigmoid();

        PreparedExecution execution = CompiledGraph.compile(out, fuseOnlyInferenceConfig())
                .prepare(new RuntimeConfig(
                        config.backend.KernelTuningConfig.defaultsInference(),
                        config.runtime.ApproximationConfig.defaults(),
                        config.runtime.BlasConfig.disabled(),
                        new FusedExecutionPolicy(FusedPrimaryBackend.ASM, true)
                ));

        var fusedStep = execution.forwardSteps().stream()
                .filter(step -> step.metadata().fusedExecutable() != null)
                .findFirst()
                .orElseThrow();

        assertTrue(fusedStep.metadata().fusedExecutable().getClass().getName().startsWith("backend.cpu.fused.asm."));
    }

    @Test
    void bfloat16LinearToReluPublishesFloatContinuationInInference() {
        Tensor input = new Tensor(new double[32 * 64], new int[]{32, 64}, null, "input", DataType.BFLOAT16);
        Tensor weight = new Tensor(new double[64 * 96], new int[]{64, 96}, null, "weight", DataType.BFLOAT16);
        Tensor bias = new Tensor(new double[96], new int[]{96}, null, "bias", DataType.BFLOAT16);
        Tensor out = input.linear(weight, bias).relu();

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults())
                .prepare(bfloat16BlasRuntime());

        var linearStep = execution.forwardSteps().stream()
                .filter(step -> step.node().getOperation() != null && step.node().getOperation().opType() == Operation.OpType.LINEAR)
                .findFirst()
                .orElseThrow();

        assertEquals("BFLOAT16", linearStep.metadata().cpuPlan().computeContract().storageType().name());
        assertEquals("F32", linearStep.metadata().cpuPlan().computeContract().computeType().name());
        assertEquals("CPU_MATMUL_BLAS", linearStep.metadata().cpuPlan().computeContract().backend().name());
        assertTrue(linearStep.metadata().cpuPlan().publishFloatContinuation());
    }

    @Test
    void bfloat16MatmulToAddPublishesFloatContinuationInInference() {
        Tensor a = new Tensor(new double[64 * 64], new int[]{64, 64}, null, "a", DataType.BFLOAT16);
        Tensor b = new Tensor(new double[64 * 96], new int[]{64, 96}, null, "b", DataType.BFLOAT16);
        Tensor c = new Tensor(new double[64 * 96], new int[]{64, 96}, null, "c", DataType.BFLOAT16);
        Tensor out = a.matmul(b).add(c);

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults())
                .prepare(bfloat16BlasRuntime());

        var matmulStep = execution.forwardSteps().stream()
                .filter(step -> step.node().getOperation() != null && step.node().getOperation().opType() == Operation.OpType.MATMUL)
                .findFirst()
                .orElseThrow();

        assertEquals("BFLOAT16", matmulStep.metadata().cpuPlan().computeContract().storageType().name());
        assertEquals("F32", matmulStep.metadata().cpuPlan().computeContract().computeType().name());
        assertEquals("CPU_MATMUL_BLAS", matmulStep.metadata().cpuPlan().computeContract().backend().name());
        assertTrue(matmulStep.metadata().cpuPlan().publishFloatContinuation());
        assertNotNull(matmulStep.metadata().cpuPlan().matMulExecutable());
        assertEquals("BF16BlasMatMulExecutable", matmulStep.metadata().cpuPlan().matMulExecutable().getClass().getSimpleName());
    }

    @Test
    void bfloat16MatmulToBroadcastAddToReluPublishesFloatContinuationInInference() {
        Tensor a = new Tensor(new double[16 * 8], new int[]{16, 8}, null, "a", DataType.BFLOAT16);
        Tensor b = new Tensor(new double[8 * 12], new int[]{8, 12}, null, "b", DataType.BFLOAT16);
        Tensor bias = new Tensor(new double[12], new int[]{12}, null, "bias", DataType.BFLOAT16);
        Tensor out = a.matmul(b).add(bias).relu();

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .prepare(RuntimeConfig.inferenceDefaults());

        var matmulStep = execution.forwardSteps().stream()
                .filter(step -> step.node().getOperation() != null && step.node().getOperation().opType() == Operation.OpType.MATMUL)
                .findFirst()
                .orElseThrow();
        var addStep = execution.forwardSteps().stream()
                .filter(step -> step.node().getOperation() != null && step.node().getOperation().opType() == Operation.OpType.ADD)
                .findFirst()
                .orElseThrow();

        assertTrue(matmulStep.metadata().cpuPlan().publishFloatContinuation());
        assertTrue(addStep.metadata().cpuPlan().publishFloatContinuation());
    }

    @Test
    void float64MatmulPrepareBuildsBlasExecutableWhenEligible() {
        Tensor a = new Tensor(new double[64 * 64], new int[]{64, 64}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[64 * 96], new int[]{64, 96}, null, "b", DataType.FLOAT64);
        Tensor out = a.matmul(b);

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .prepare(bfloat16BlasRuntime());

        var matmulStep = execution.forwardSteps().stream()
                .filter(step -> step.node().getOperation() != null && step.node().getOperation().opType() == Operation.OpType.MATMUL)
                .findFirst()
                .orElseThrow();

        assertNotNull(matmulStep.metadata().cpuPlan().matMulExecutable());
        assertEquals("F64BlasMatMulExecutable", matmulStep.metadata().cpuPlan().matMulExecutable().getClass().getSimpleName());
    }

    @Test
    void float32MatmulPrepareBuildsJavaExecutableWhenBlasDisabled() {
        Tensor a = new Tensor(new float[64 * 64], new int[]{64, 64}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[64 * 96], new int[]{64, 96}, null, "b", DataType.FLOAT32);
        Tensor out = a.matmul(b);

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .prepare(RuntimeConfig.inferenceDefaults());

        var matmulStep = execution.forwardSteps().stream()
                .filter(step -> step.node().getOperation() != null && step.node().getOperation().opType() == Operation.OpType.MATMUL)
                .findFirst()
                .orElseThrow();

        assertNotNull(matmulStep.metadata().cpuPlan().matMulExecutable());
        assertEquals("F32JavaMatMulExecutable", matmulStep.metadata().cpuPlan().matMulExecutable().getClass().getSimpleName());
    }

    @Test
    void bfloat16MatmulToFusedNumericChainPublishesFloatContinuationInInference() {
        Tensor a = new Tensor(new double[64 * 64], new int[]{64, 64}, null, "a", DataType.BFLOAT16);
        Tensor b = new Tensor(new double[64 * 96], new int[]{64, 96}, null, "b", DataType.BFLOAT16);
        Tensor out = a.matmul(b).relu().abs().clampMax(1.0);

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults())
                .prepare(bfloat16BlasRuntime());

        var matmulStep = execution.forwardSteps().stream()
                .filter(step -> step.node().getOperation() != null && step.node().getOperation().opType() == Operation.OpType.MATMUL)
                .findFirst()
                .orElseThrow();

        assertEquals("BFLOAT16", matmulStep.metadata().cpuPlan().computeContract().storageType().name());
        assertEquals("F32", matmulStep.metadata().cpuPlan().computeContract().computeType().name());
        assertEquals("CPU_MATMUL_BLAS", matmulStep.metadata().cpuPlan().computeContract().backend().name());
        assertTrue(matmulStep.metadata().cpuPlan().publishFloatContinuation());
    }

    @Test
    void bfloat16MatmulToNegPublishesFloatContinuationInInference() {
        Tensor a = new Tensor(new double[64 * 64], new int[]{64, 64}, null, "a", DataType.BFLOAT16);
        Tensor b = new Tensor(new double[64 * 96], new int[]{64, 96}, null, "b", DataType.BFLOAT16);
        Tensor out = a.matmul(b).neg();

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults())
                .prepare(bfloat16BlasRuntime());

        var matmulStep = execution.forwardSteps().stream()
                .filter(step -> step.node().getOperation() != null && step.node().getOperation().opType() == Operation.OpType.MATMUL)
                .findFirst()
                .orElseThrow();

        assertTrue(matmulStep.metadata().cpuPlan().publishFloatContinuation());
    }

    @Test
    void bfloat16MatmulToMulScalarPublishesFloatContinuationInInference() {
        Tensor a = new Tensor(new double[64 * 64], new int[]{64, 64}, null, "a", DataType.BFLOAT16);
        Tensor b = new Tensor(new double[64 * 96], new int[]{64, 96}, null, "b", DataType.BFLOAT16);
        Tensor out = a.matmul(b).mul(0.5);

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults())
                .prepare(bfloat16BlasRuntime());

        var matmulStep = execution.forwardSteps().stream()
                .filter(step -> step.node().getOperation() != null && step.node().getOperation().opType() == Operation.OpType.MATMUL)
                .findFirst()
                .orElseThrow();

        assertTrue(matmulStep.metadata().cpuPlan().publishFloatContinuation());
    }

    @Test
    void bfloat16MatmulToPowPublishesFloatContinuationInInference() {
        Tensor a = new Tensor(new double[64 * 64], new int[]{64, 64}, null, "a", DataType.BFLOAT16);
        Tensor b = new Tensor(new double[64 * 96], new int[]{64, 96}, null, "b", DataType.BFLOAT16);
        Tensor out = a.matmul(b).pow(1.5);

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults())
                .prepare(bfloat16BlasRuntime());

        var matmulStep = execution.forwardSteps().stream()
                .filter(step -> step.node().getOperation() != null && step.node().getOperation().opType() == Operation.OpType.MATMUL)
                .findFirst()
                .orElseThrow();

        assertTrue(matmulStep.metadata().cpuPlan().publishFloatContinuation());
    }

    @Test
    void bfloat16MatmulReshapeToNegPublishesFloatContinuationInInference() {
        Tensor a = new Tensor(new double[64 * 64], new int[]{64, 64}, null, "a", DataType.BFLOAT16);
        Tensor b = new Tensor(new double[64 * 96], new int[]{64, 96}, null, "b", DataType.BFLOAT16);
        Tensor out = a.matmul(b).reshape(32, 192).neg();

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults())
                .prepare(bfloat16BlasRuntime());

        var matmulStep = execution.forwardSteps().stream()
                .filter(step -> step.node().getOperation() != null && step.node().getOperation().opType() == Operation.OpType.MATMUL)
                .findFirst()
                .orElseThrow();

        assertTrue(matmulStep.metadata().cpuPlan().publishFloatContinuation());
    }

    @Test
    void bfloat16MatmulToWhereToReluPublishesFloatContinuationInInference() {
        Tensor a = new Tensor(new double[8 * 8], new int[]{8, 8}, null, "a", DataType.BFLOAT16);
        Tensor b = new Tensor(new double[8 * 12], new int[]{8, 12}, null, "b", DataType.BFLOAT16);
        Tensor mask = new Tensor(new byte[8], new int[]{8, 1}, null, "mask", DataType.BOOL);
        Tensor fill = Tensor.scalar(-1.0, DataType.BFLOAT16);
        Tensor out = Tensor.where(mask, a.matmul(b), fill).relu();

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .prepare(RuntimeConfig.inferenceDefaults());

        var matmulStep = execution.forwardSteps().stream()
                .filter(step -> step.node().getOperation() != null && step.node().getOperation().opType() == Operation.OpType.MATMUL)
                .findFirst()
                .orElseThrow();
        var whereStep = execution.forwardSteps().stream()
                .filter(step -> step.node().getOperation() != null && step.node().getOperation().opType() == Operation.OpType.WHERE)
                .findFirst()
                .orElseThrow();

        assertTrue(matmulStep.metadata().cpuPlan().publishFloatContinuation());
        assertTrue(whereStep.metadata().cpuPlan().publishFloatContinuation());
    }

    @Test
    void gpuMetalPartitionPrepareBuildsSingleAnchorStepForMatmulAddReluChain() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{3, 2}, null, "b", DataType.FLOAT32);
        Tensor bias = new Tensor(new float[]{1f, -1f}, new int[]{2}, null, "bias", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor add = matmul.add(bias);
        Tensor out = add.relu();

        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(add, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .toList();
        assertEquals(1, gpuSteps.size());
        assertEquals(1, execution.prepareTrace().backendSelection().selectedCount());
        var anchor = gpuSteps.getFirst();
        assertEquals(ComputeBackend.GPU_METAL, anchor.metadata().backend());
        assertEquals(PartitionExecutionRole.ANCHOR, anchor.metadata().partitionRole());
        assertNotNull(anchor.metadata().acceleratorExecutable());
        assertTrue(anchor.metadata().acceleratorExecutable() instanceof PreparedMetalExecutable);
        PreparedMetalExecutable executable = (PreparedMetalExecutable) anchor.metadata().acceleratorExecutable();
        assertNotNull(executable.bridgeContext());
        assertNotNull(executable.bridgeExecutable());
    }

    @Test
    void gpuMetalMockPartitionExecutionMatchesCpuForMatmulAddReluChain() {
        Tensor cpuA = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "cpuA", DataType.FLOAT32);
        Tensor cpuB = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{3, 2}, null, "cpuB", DataType.FLOAT32);
        Tensor cpuBias = new Tensor(new float[]{1f, -1f}, new int[]{2}, null, "cpuBias", DataType.FLOAT32);
        Tensor cpuOut = cpuA.matmul(cpuB).add(cpuBias).relu();

        CompiledGraph.compile(cpuOut, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{3, 2}, null, "b", DataType.FLOAT32);
        Tensor bias = new Tensor(new float[]{1f, -1f}, new int[]{2}, null, "bias", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor add = matmul.add(bias);
        Tensor out = add.relu();

        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(add, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 1e-6);
    }

    @Test
    void gpuMetalSingleMatmulCanExecuteThroughExplicitAppleShim() {
        String explicitLib = System.getProperty("synaptik.metal.mps.lib");
        assumeTrue(explicitLib != null && !explicitLib.isBlank());

        Tensor cpuA = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "cpuA", DataType.FLOAT32);
        Tensor cpuB = new Tensor(new float[]{7f, 8f, 9f, 10f, 11f, 12f}, new int[]{3, 2}, null, "cpuB", DataType.FLOAT32);
        Tensor cpuOut = cpuA.matmul(cpuB);
        CompiledGraph.compile(cpuOut, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{7f, 8f, 9f, 10f, 11f, 12f}, new int[]{3, 2}, null, "b", DataType.FLOAT32);
        Tensor out = a.matmul(b);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .toList();
        assertEquals(1, gpuSteps.size());
        PreparedMetalExecutable executable = (PreparedMetalExecutable) gpuSteps.getFirst().metadata().acceleratorExecutable();
        assumeTrue(executable.bridgeContext().available());
        assumeTrue(executable.bridgeExecutable().available());

        execution.execute(ExecutionMode.FORWARD);

        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 1e-5);
    }

    @Test
    void gpuMetalDirectSdpaFallsBackToCpuUntilNativeScaleContractMatchesCpu() {
        Tensor cpuQ = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "cpuQ", DataType.FLOAT32);
        Tensor cpuK = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "cpuK", DataType.FLOAT32);
        Tensor cpuV = new Tensor(new float[]{10f, 1f, 1f, 10f}, new int[]{1, 2, 2}, null, "cpuV", DataType.FLOAT32);
        Tensor cpuOut = cpuQ.scaledDotProductAttention(cpuK, cpuV, tensor.options.AttentionOptions.defaults().withScale(0.5));
        CompiledGraph.compile(cpuOut, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor q = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "q", DataType.FLOAT32);
        Tensor k = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "k", DataType.FLOAT32);
        Tensor v = new Tensor(new float[]{10f, 1f, 1f, 10f}, new int[]{1, 2, 2}, null, "v", DataType.FLOAT32);
        Tensor out = q.scaledDotProductAttention(k, v, tensor.options.AttentionOptions.defaults().withScale(0.5));
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .prepare(RuntimeConfig.inferenceDefaults());
        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .toList();
        assertTrue(gpuSteps.isEmpty());

        execution.execute(ExecutionMode.FORWARD);

        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 1e-5);
    }

    @Test
    void gpuMetalDirectMaskedSdpaFallsBackToCpuBecauseNativeMaskContractDiffers() {
        Tensor cpuQ = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "cpuQMask", DataType.FLOAT32);
        Tensor cpuK = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "cpuKMask", DataType.FLOAT32);
        Tensor cpuV = new Tensor(new float[]{10f, 1f, 1f, 10f}, new int[]{1, 2, 2}, null, "cpuVMask", DataType.FLOAT32);
        Tensor cpuMask = new Tensor(new byte[]{1, 0, 1, 1}, new int[]{1, 2, 2}, null, "cpuMask", DataType.BOOL);
        Tensor cpuOut = cpuQ.scaledDotProductAttention(cpuK, cpuV, cpuMask, tensor.options.AttentionOptions.defaults().withScale(0.5));
        CompiledGraph.compile(cpuOut, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor q = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "qMask", DataType.FLOAT32);
        Tensor k = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "kMask", DataType.FLOAT32);
        Tensor v = new Tensor(new float[]{10f, 1f, 1f, 10f}, new int[]{1, 2, 2}, null, "vMask", DataType.FLOAT32);
        Tensor mask = new Tensor(new byte[]{1, 0, 1, 1}, new int[]{1, 2, 2}, null, "mask", DataType.BOOL);
        Tensor out = q.scaledDotProductAttention(k, v, mask, tensor.options.AttentionOptions.defaults().withScale(0.5));
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .prepare(RuntimeConfig.inferenceDefaults());
        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .toList();
        assertTrue(gpuSteps.isEmpty());

        execution.execute(ExecutionMode.FORWARD);

        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 1e-5);
    }

    @Test
    void gpuMetalMatmulWithTransposedRhsCanExecuteThroughExplicitAppleShim() {
        String explicitLib = System.getProperty("synaptik.metal.mps.lib");
        assumeTrue(explicitLib != null && !explicitLib.isBlank());

        Tensor cpuA = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "cpuA", DataType.FLOAT32);
        Tensor cpuB = new Tensor(new float[]{7f, 8f, 9f, 10f, 11f, 12f}, new int[]{2, 3}, null, "cpuB", DataType.FLOAT32);
        Tensor cpuOut = cpuA.matmul(cpuB.transpose());
        CompiledGraph.compile(cpuOut, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{7f, 8f, 9f, 10f, 11f, 12f}, new int[]{2, 3}, null, "b", DataType.FLOAT32);
        Tensor out = a.matmul(b.transpose());
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .toList();
        assertEquals(1, gpuSteps.size());
        PreparedMetalExecutable executable = (PreparedMetalExecutable) gpuSteps.getFirst().metadata().acceleratorExecutable();
        assumeTrue(executable.bridgeContext().available());
        assumeTrue(executable.bridgeExecutable().available());

        execution.execute(ExecutionMode.FORWARD);

        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 1e-5);
    }

    @Test
    void gpuMetalPureElementwiseChainCanPrepareAndExecute() {
        Tensor cpuA = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{4}, null, "cpuA", DataType.FLOAT32);
        Tensor cpuB = new Tensor(new float[]{5f, 6f, 7f, 8f}, new int[]{4}, null, "cpuB", DataType.FLOAT32);
        Tensor cpuOut = cpuA.add(cpuB).relu().exp();
        CompiledGraph.compile(cpuOut, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{4}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{5f, 6f, 7f, 8f}, new int[]{4}, null, "b", DataType.FLOAT32);
        Tensor add = a.add(b);
        Tensor relu = add.relu();
        Tensor out = relu.exp();

        TensorInternalAccess.setBackend(add, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(relu, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .toList();
        assertEquals(1, gpuSteps.size());
        PreparedMetalExecutable executable = (PreparedMetalExecutable) gpuSteps.getFirst().metadata().acceleratorExecutable();
        assertEquals(3, executable.plan().lowering().dagSpec().nodes().size());

        execution.execute(ExecutionMode.FORWARD);

        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 3e-3);
    }

    @Test
    void gpuMetalPureElementwiseChainUsesFusedElementwiseLoweringWhenOptimizerRegionsExist() {
        Tensor cpuA = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{4}, null, "cpuAOpt", DataType.FLOAT32);
        Tensor cpuB = new Tensor(new float[]{5f, 6f, 7f, 8f}, new int[]{4}, null, "cpuBOpt", DataType.FLOAT32);
        Tensor cpuOut = cpuA.add(cpuB).relu().exp();
        CompiledGraph.compile(cpuOut, OptimizerConfig.inferenceDefaults())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{4}, null, "aOpt", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{5f, 6f, 7f, 8f}, new int[]{4}, null, "bOpt", DataType.FLOAT32);
        Tensor add = a.add(b);
        Tensor relu = add.relu();
        Tensor out = relu.exp();

        TensorInternalAccess.setBackend(add, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(relu, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults())
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .toList();
        assertEquals(1, gpuSteps.size());
        PreparedMetalExecutable executable = (PreparedMetalExecutable) gpuSteps.getFirst().metadata().acceleratorExecutable();
        assertEquals(backend.lowering.LoweringFamily.METAL_FUSED_ELEMENTWISE_GRAPH, executable.loweringFamily());

        execution.execute(ExecutionMode.FORWARD);

        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 3e-3);
    }

    @Test
    void gpuCudaPureElementwiseChainCanPrepareAndExecuteThroughFallback() {
        Tensor cpuA = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{4}, null, "cpuCudaA", DataType.FLOAT32);
        Tensor cpuB = new Tensor(new float[]{5f, 6f, 7f, 8f}, new int[]{4}, null, "cpuCudaB", DataType.FLOAT32);
        Tensor cpuOut = cpuA.add(cpuB).relu().exp();
        CompiledGraph.compile(cpuOut, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{4}, null, "cudaA", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{5f, 6f, 7f, 8f}, new int[]{4}, null, "cudaB", DataType.FLOAT32);
        Tensor add = a.add(b);
        Tensor relu = add.relu();
        Tensor out = relu.exp();

        TensorInternalAccess.setBackend(add, ComputeBackend.GPU_CUDA);
        TensorInternalAccess.setBackend(relu, ComputeBackend.GPU_CUDA);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_CUDA);

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults())
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_CUDA)
                .toList();
        assertEquals(1, gpuSteps.size());
        PreparedCudaExecutable executable = (PreparedCudaExecutable) gpuSteps.getFirst().metadata().acceleratorExecutable();
        assertEquals(backend.lowering.LoweringFamily.CUDA_FUSED_ELEMENTWISE_GRAPH, executable.loweringFamily());

        execution.execute(ExecutionMode.FORWARD);

        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 1e-5);
    }

    @Test
    void gpuMetalBackwardMatmulCanPrepareAndMatchCpuGradients() {
        String explicitLib = System.getProperty("synaptik.metal.mps.lib");
        assumeTrue(explicitLib != null && !explicitLib.isBlank());

        Tensor cpuA = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "cpuA", DataType.FLOAT32);
        Tensor cpuB = new Tensor(new float[]{7f, 8f, 9f, 10f, 11f, 12f}, new int[]{3, 2}, null, "cpuB", DataType.FLOAT32);
        cpuA.setRequiresGrad(true);
        cpuB.setRequiresGrad(true);
        Tensor cpuMatmul = cpuA.matmul(cpuB);
        Tensor cpuLoss = cpuMatmul.sum();
        CompiledGraph.compile(cpuLoss, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);
        double[] expectedGradA = cpuA.getGradient().toDoubleArrayCopy().clone();
        double[] expectedGradB = cpuB.getGradient().toDoubleArrayCopy().clone();

        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{7f, 8f, 9f, 10f, 11f, 12f}, new int[]{3, 2}, null, "b", DataType.FLOAT32);
        a.setRequiresGrad(true);
        b.setRequiresGrad(true);
        Tensor matmul = a.matmul(b);
        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_METAL);
        Tensor loss = matmul.sum();

        PreparedExecution execution = CompiledGraph.compile(loss, OptimizerConfig.noOptimization())
                .prepare(RuntimeConfig.trainingDefaults());

        long gpuBackwardMatmuls = execution.backwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .count();
        assertTrue(gpuBackwardMatmuls >= 1);

        execution.execute(ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(expectedGradA, a.getGradient().toDoubleArrayCopy(), 1e-5);
        assertArrayEquals(expectedGradB, b.getGradient().toDoubleArrayCopy(), 1e-5);
    }

    @Test
    void gpuMetalBackwardSoftmaxGradCanPrepareAndMatchCpuGradients() {
        Tensor cpuInput = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "cpuInput", DataType.FLOAT32);
        cpuInput.setRequiresGrad(true);
        Tensor cpuSoftmax = cpuInput.exp().softmax(1);
        Tensor cpuLoss = cpuSoftmax.sum();
        CompiledGraph.compile(cpuLoss, OptimizerConfig.trainingDefaults())
                .execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);
        double[] expectedGrad = cpuInput.getGradient().toDoubleArrayCopy().clone();

        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "input", DataType.FLOAT32);
        input.setRequiresGrad(true);
        Tensor softmax = input.exp().softmax(1);
        TensorInternalAccess.setBackend(softmax, ComputeBackend.GPU_METAL);
        Tensor loss = softmax.sum();

        PreparedExecution execution = CompiledGraph.compile(loss, OptimizerConfig.trainingDefaults())
                .prepare(RuntimeConfig.trainingDefaults());

        var gpuSoftmaxGrad = execution.backwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .filter(step -> step.node().getOperation() != null && step.node().getOperation().opType() == Operation.OpType.SOFTMAX_GRAD)
                .findFirst()
                .orElseThrow();
        assertEquals(ComputeBackend.GPU_METAL, gpuSoftmaxGrad.metadata().backend());

        execution.execute(ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(expectedGrad, input.getGradient().toDoubleArrayCopy(), 1e-5);
    }

    @Test
    void gpuMetalBackwardLogSoftmaxGradCanPrepareAndMatchCpuGradients() {
        Tensor cpuInput = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "cpuInput", DataType.FLOAT32);
        cpuInput.setRequiresGrad(true);
        Tensor cpuLogSoftmax = cpuInput.exp().logSoftmax(1);
        Tensor cpuLoss = cpuLogSoftmax.sum();
        CompiledGraph.compile(cpuLoss, OptimizerConfig.trainingDefaults())
                .execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);
        double[] expectedGrad = cpuInput.getGradient().toDoubleArrayCopy().clone();

        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "input", DataType.FLOAT32);
        input.setRequiresGrad(true);
        Tensor logSoftmax = input.exp().logSoftmax(1);
        TensorInternalAccess.setBackend(logSoftmax, ComputeBackend.GPU_METAL);
        Tensor loss = logSoftmax.sum();

        PreparedExecution execution = CompiledGraph.compile(loss, OptimizerConfig.trainingDefaults())
                .prepare(RuntimeConfig.trainingDefaults());

        var gpuLogSoftmaxGrad = execution.backwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .filter(step -> step.node().getOperation() != null && step.node().getOperation().opType() == Operation.OpType.LOG_SOFTMAX_GRAD)
                .findFirst()
                .orElseThrow();
        assertEquals(ComputeBackend.GPU_METAL, gpuLogSoftmaxGrad.metadata().backend());

        execution.execute(ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(expectedGrad, input.getGradient().toDoubleArrayCopy(), 5e-5);
    }

    @Test
    void gpuMetalBackwardReduceMinGradCanPrepareAndMatchCpuGradients() {
        Tensor cpuInput = new Tensor(new float[]{
                1f, 1f, 2f,
                3f, 2f, 2f
        }, new int[]{2, 3}, null, "cpuInput", DataType.FLOAT32);
        cpuInput.setRequiresGrad(true);
        Tensor cpuReduced = cpuInput.min(1, true);
        Tensor cpuLoss = cpuReduced.sum();
        CompiledGraph.compile(cpuLoss, OptimizerConfig.trainingDefaults())
                .execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);
        double[] expectedGrad = cpuInput.getGradient().toDoubleArrayCopy().clone();

        Tensor input = new Tensor(new float[]{
                1f, 1f, 2f,
                3f, 2f, 2f
        }, new int[]{2, 3}, null, "input", DataType.FLOAT32);
        input.setRequiresGrad(true);
        Tensor reduced = input.min(1, true);
        TensorInternalAccess.setBackend(reduced, ComputeBackend.GPU_METAL);
        Tensor loss = reduced.sum();

        PreparedExecution execution = CompiledGraph.compile(loss, OptimizerConfig.trainingDefaults())
                .prepare(RuntimeConfig.trainingDefaults());

        var gpuReduceMinGrad = execution.backwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .filter(step -> step.node().getOperation() != null && step.node().getOperation().opType() == Operation.OpType.REDUCE_MIN_GRAD)
                .findFirst()
                .orElseThrow();
        assertEquals(ComputeBackend.GPU_METAL, gpuReduceMinGrad.metadata().backend());

        execution.execute(ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(expectedGrad, input.getGradient().toDoubleArrayCopy(), 1e-5);
    }

    @Test
    void gpuMetalBackwardReduceMaxGradCanPrepareAndMatchCpuGradients() {
        Tensor cpuInput = new Tensor(new float[]{1f, 5f, 5f, 2f}, new int[]{4}, null, "cpuInput", DataType.FLOAT32);
        cpuInput.setRequiresGrad(true);
        Tensor cpuReduced = cpuInput.max();
        Tensor cpuLoss = cpuReduced.sum();
        CompiledGraph.compile(cpuLoss, OptimizerConfig.trainingDefaults())
                .execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);
        double[] expectedGrad = cpuInput.getGradient().toDoubleArrayCopy().clone();

        Tensor input = new Tensor(new float[]{1f, 5f, 5f, 2f}, new int[]{4}, null, "input", DataType.FLOAT32);
        input.setRequiresGrad(true);
        Tensor reduced = input.max();
        TensorInternalAccess.setBackend(reduced, ComputeBackend.GPU_METAL);
        Tensor loss = reduced.sum();

        PreparedExecution execution = CompiledGraph.compile(loss, OptimizerConfig.trainingDefaults())
                .prepare(RuntimeConfig.trainingDefaults());

        var gpuReduceMaxGrad = execution.backwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .filter(step -> step.node().getOperation() != null && step.node().getOperation().opType() == Operation.OpType.REDUCE_MAX_GRAD)
                .findFirst()
                .orElseThrow();
        assertEquals(ComputeBackend.GPU_METAL, gpuReduceMaxGrad.metadata().backend());

        execution.execute(ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(expectedGrad, input.getGradient().toDoubleArrayCopy(), 1e-5);
    }

    @Test
    void gpuMetalBackwardMinGradCanPrepareAndMatchCpuGradients() {
        Tensor cpuA = new Tensor(new float[]{1f, 5f, 3f}, new int[]{3}, null, "cpuA", DataType.FLOAT32);
        Tensor cpuB = new Tensor(new float[]{2f, 4f, 3f}, new int[]{3}, null, "cpuB", DataType.FLOAT32);
        cpuA.setRequiresGrad(true);
        cpuB.setRequiresGrad(true);
        Tensor cpuLoss = cpuA.min(cpuB).sum();
        CompiledGraph.compile(cpuLoss, OptimizerConfig.trainingDefaults())
                .execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);
        double[] expectedGradA = cpuA.getGradient().toDoubleArrayCopy().clone();
        double[] expectedGradB = cpuB.getGradient().toDoubleArrayCopy().clone();

        Tensor a = new Tensor(new float[]{1f, 5f, 3f}, new int[]{3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{2f, 4f, 3f}, new int[]{3}, null, "b", DataType.FLOAT32);
        a.setRequiresGrad(true);
        b.setRequiresGrad(true);
        Tensor min = a.min(b);
        TensorInternalAccess.setBackend(min, ComputeBackend.GPU_METAL);
        Tensor loss = min.sum();

        PreparedExecution execution = CompiledGraph.compile(loss, OptimizerConfig.trainingDefaults())
                .prepare(RuntimeConfig.trainingDefaults());

        var gpuMinGrad = execution.backwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .filter(step -> step.node().getOperation() != null && step.node().getOperation().opType() == Operation.OpType.MIN_GRAD)
                .findFirst()
                .orElseThrow();
        assertEquals(ComputeBackend.GPU_METAL, gpuMinGrad.metadata().backend());

        execution.execute(ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(expectedGradA, a.getGradient().toDoubleArrayCopy(), 1e-5);
        assertArrayEquals(expectedGradB, b.getGradient().toDoubleArrayCopy(), 1e-5);
    }

    @Test
    void gpuMetalBackwardMaxGradCanPrepareAndMatchCpuGradients() {
        Tensor cpuA = new Tensor(new float[]{1f, 5f, 3f}, new int[]{3}, null, "cpuA", DataType.FLOAT32);
        Tensor cpuB = new Tensor(new float[]{2f, 4f, 3f}, new int[]{3}, null, "cpuB", DataType.FLOAT32);
        cpuA.setRequiresGrad(true);
        cpuB.setRequiresGrad(true);
        Tensor cpuLoss = cpuA.max(cpuB).sum();
        CompiledGraph.compile(cpuLoss, OptimizerConfig.trainingDefaults())
                .execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);
        double[] expectedGradA = cpuA.getGradient().toDoubleArrayCopy().clone();
        double[] expectedGradB = cpuB.getGradient().toDoubleArrayCopy().clone();

        Tensor a = new Tensor(new float[]{1f, 5f, 3f}, new int[]{3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{2f, 4f, 3f}, new int[]{3}, null, "b", DataType.FLOAT32);
        a.setRequiresGrad(true);
        b.setRequiresGrad(true);
        Tensor max = a.max(b);
        TensorInternalAccess.setBackend(max, ComputeBackend.GPU_METAL);
        Tensor loss = max.sum();

        PreparedExecution execution = CompiledGraph.compile(loss, OptimizerConfig.trainingDefaults())
                .prepare(RuntimeConfig.trainingDefaults());

        var gpuMaxGrad = execution.backwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .filter(step -> step.node().getOperation() != null && step.node().getOperation().opType() == Operation.OpType.MAX_GRAD)
                .findFirst()
                .orElseThrow();
        assertEquals(ComputeBackend.GPU_METAL, gpuMaxGrad.metadata().backend());

        execution.execute(ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(expectedGradA, a.getGradient().toDoubleArrayCopy(), 1e-5);
        assertArrayEquals(expectedGradB, b.getGradient().toDoubleArrayCopy(), 1e-5);
    }

    @Test
    void gpuMetalBackwardSdpaCanPrepareAndMatchCpuGradients() {
        Tensor cpuQ = new Tensor(new float[]{
                1f, 0f,
                0f, 1f
        }, new int[]{1, 2, 2}, null, "cpuQ", DataType.FLOAT32);
        Tensor cpuK = new Tensor(new float[]{
                1f, 0f,
                0f, 1f
        }, new int[]{1, 2, 2}, null, "cpuK", DataType.FLOAT32);
        Tensor cpuV = new Tensor(new float[]{
                10f, 1f,
                1f, 10f
        }, new int[]{1, 2, 2}, null, "cpuV", DataType.FLOAT32);
        cpuQ.setRequiresGrad(true);
        cpuK.setRequiresGrad(true);
        cpuV.setRequiresGrad(true);
        Tensor cpuLoss = cpuQ.scaledDotProductAttention(cpuK, cpuV, tensor.options.AttentionOptions.defaults().withScale(1.0)).sum();
        CompiledGraph.compile(cpuLoss, OptimizerConfig.trainingDefaults())
                .execute(RuntimeConfig.trainingDefaults(), ExecutionMode.FORWARD_BACKWARD);
        double[] expectedGradQ = cpuQ.getGradient().toDoubleArrayCopy().clone();
        double[] expectedGradK = cpuK.getGradient().toDoubleArrayCopy().clone();
        double[] expectedGradV = cpuV.getGradient().toDoubleArrayCopy().clone();

        Tensor q = new Tensor(new float[]{
                1f, 0f,
                0f, 1f
        }, new int[]{1, 2, 2}, null, "q", DataType.FLOAT32);
        Tensor k = new Tensor(new float[]{
                1f, 0f,
                0f, 1f
        }, new int[]{1, 2, 2}, null, "k", DataType.FLOAT32);
        Tensor v = new Tensor(new float[]{
                10f, 1f,
                1f, 10f
        }, new int[]{1, 2, 2}, null, "v", DataType.FLOAT32);
        q.setRequiresGrad(true);
        k.setRequiresGrad(true);
        v.setRequiresGrad(true);
        Tensor attention = q.scaledDotProductAttention(k, v, tensor.options.AttentionOptions.defaults().withScale(1.0));
        TensorInternalAccess.setBackend(attention, ComputeBackend.GPU_METAL);
        Tensor loss = attention.sum();

        PreparedExecution execution = CompiledGraph.compile(loss, OptimizerConfig.trainingDefaults())
                .prepare(RuntimeConfig.trainingDefaults());

        long gpuBackwardSdpa = execution.backwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .filter(step -> step.node().getOperation() != null && step.node().getOperation().opType() == Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION_BACKWARD)
                .count();
        assertTrue(gpuBackwardSdpa >= 3);

        execution.execute(ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(expectedGradQ, q.getGradient().toDoubleArrayCopy(), 1e-5);
        assertArrayEquals(expectedGradK, k.getGradient().toDoubleArrayCopy(), 1e-5);
        assertArrayEquals(expectedGradV, v.getGradient().toDoubleArrayCopy(), 1e-5);
    }


    @Test
    void gpuMetalLinearBiasTanhCanExecuteThroughExplicitAppleShim() {
        String explicitLib = System.getProperty("synaptik.metal.mps.lib");
        assumeTrue(explicitLib != null && !explicitLib.isBlank());

        Tensor cpuInput = new Tensor(new float[]{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f}, new int[]{2, 3}, null, "cpuInput", DataType.FLOAT32);
        Tensor cpuWeight = new Tensor(new float[]{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f}, new int[]{3, 2}, null, "cpuWeight", DataType.FLOAT32);
        Tensor cpuBias = new Tensor(new float[]{0.1f, 0.2f}, new int[]{2}, null, "cpuBias", DataType.FLOAT32);
        Tensor cpuOut = cpuInput.linear(cpuWeight, cpuBias).tanh();
        CompiledGraph.compile(cpuOut, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor input = new Tensor(new float[]{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f}, new int[]{2, 3}, null, "input", DataType.FLOAT32);
        Tensor weight = new Tensor(new float[]{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f}, new int[]{3, 2}, null, "weight", DataType.FLOAT32);
        Tensor bias = new Tensor(new float[]{0.1f, 0.2f}, new int[]{2}, null, "bias", DataType.FLOAT32);
        Tensor linear = input.linear(weight, bias);
        Tensor out = linear.tanh();

        TensorInternalAccess.setBackend(linear, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .toList();
        assertEquals(1, gpuSteps.size());
        PreparedMetalExecutable executable = (PreparedMetalExecutable) gpuSteps.getFirst().metadata().acceleratorExecutable();
        assumeTrue(executable.bridgeContext().available());
        assumeTrue(executable.bridgeExecutable().available());

        execution.execute(ExecutionMode.FORWARD);

        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 1e-5);
    }

    @Test
    void gpuMetalPreparedExecutableReusesCompiledHandleForSameSubgraphSignature() {
        String explicitLib = System.getProperty("synaptik.metal.mps.lib");
        assumeTrue(explicitLib != null && !explicitLib.isBlank());

        Tensor a1 = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "a1", DataType.FLOAT32);
        Tensor b1 = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{3, 2}, null, "b1", DataType.FLOAT32);
        Tensor bias1 = new Tensor(new float[]{1f, -1f}, new int[]{2}, null, "bias1", DataType.FLOAT32);
        Tensor matmul1 = a1.matmul(b1);
        Tensor add1 = matmul1.add(bias1);
        Tensor out1 = add1.relu();
        TensorInternalAccess.setBackend(matmul1, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(add1, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out1, ComputeBackend.GPU_METAL);

        PreparedExecution execution1 = CompiledGraph.compile(out1, OptimizerConfig.noOptimization())
                .prepare(RuntimeConfig.inferenceDefaults());
        PreparedMetalExecutable executable1 = (PreparedMetalExecutable) execution1.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .findFirst()
                .orElseThrow()
                .metadata()
                .acceleratorExecutable();

        Tensor a2 = new Tensor(new float[]{2f, 3f, 4f, 5f, 6f, 7f}, new int[]{2, 3}, null, "a2", DataType.FLOAT32);
        Tensor b2 = new Tensor(new float[]{2f, 3f, 4f, 5f, 6f, 7f}, new int[]{3, 2}, null, "b2", DataType.FLOAT32);
        Tensor bias2 = new Tensor(new float[]{2f, -2f}, new int[]{2}, null, "bias2", DataType.FLOAT32);
        Tensor matmul2 = a2.matmul(b2);
        Tensor add2 = matmul2.add(bias2);
        Tensor out2 = add2.relu();
        TensorInternalAccess.setBackend(matmul2, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(add2, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out2, ComputeBackend.GPU_METAL);

        PreparedExecution execution2 = CompiledGraph.compile(out2, OptimizerConfig.noOptimization())
                .prepare(RuntimeConfig.inferenceDefaults());
        PreparedMetalExecutable executable2 = (PreparedMetalExecutable) execution2.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .findFirst()
                .orElseThrow()
                .metadata()
                .acceleratorExecutable();

        assumeTrue(executable1.bridgeExecutable().available());
        assumeTrue(executable2.bridgeExecutable().available());
        assertEquals(executable1.bridgeExecutable().handle(), executable2.bridgeExecutable().handle());
    }

    @Test
    void gpuMetalMatmulAddReluCanExecuteThroughExplicitAppleShim() {
        String explicitLib = System.getProperty("synaptik.metal.mps.lib");
        assumeTrue(explicitLib != null && !explicitLib.isBlank());

        Tensor cpuA = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "cpuA", DataType.FLOAT32);
        Tensor cpuB = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{3, 2}, null, "cpuB", DataType.FLOAT32);
        Tensor cpuBias = new Tensor(new float[]{1f, -1f}, new int[]{2}, null, "cpuBias", DataType.FLOAT32);
        Tensor cpuOut = cpuA.matmul(cpuB).add(cpuBias).relu();
        CompiledGraph.compile(cpuOut, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{3, 2}, null, "b", DataType.FLOAT32);
        Tensor bias = new Tensor(new float[]{1f, -1f}, new int[]{2}, null, "bias", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor add = matmul.add(bias);
        Tensor out = add.relu();

        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(add, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .toList();
        assertEquals(1, gpuSteps.size());
        PreparedMetalExecutable executable = (PreparedMetalExecutable) gpuSteps.getFirst().metadata().acceleratorExecutable();
        assumeTrue(executable.bridgeContext().available());
        assumeTrue(executable.bridgeExecutable().available());

        execution.execute(ExecutionMode.FORWARD);

        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 1e-5);
    }

    @Test
    void gpuMetalPartitionPrepareBuildsSingleAnchorStepForMatmulAddTanhChain() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{3, 2}, null, "b", DataType.FLOAT32);
        Tensor bias = new Tensor(new float[]{1f, -1f}, new int[]{2}, null, "bias", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor add = matmul.add(bias);
        Tensor out = add.tanh();

        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(add, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .toList();
        assertEquals(1, gpuSteps.size());
        PreparedMetalExecutable executable = (PreparedMetalExecutable) gpuSteps.getFirst().metadata().acceleratorExecutable();
        assertNotNull(executable.bridgeExecutable());
    }

    @Test
    void gpuMetalPartitionPrepareBuildsSingleAnchorStepForMatmulNegChain() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{3, 2}, null, "b", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor out = matmul.neg();

        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .toList();
        assertEquals(1, gpuSteps.size());
        assertEquals(PartitionExecutionRole.ANCHOR, gpuSteps.getFirst().metadata().partitionRole());
    }

    @Test
    void gpuMetalPartitionPrepareBuildsSingleAnchorStepForMatmulReluSqrtInvChain() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{3, 2}, null, "b", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor relu = matmul.relu();
        Tensor sqrt = relu.sqrt();
        Tensor out = sqrt.inv();

        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(relu, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(sqrt, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .toList();
        assertEquals(1, gpuSteps.size());
        PreparedMetalExecutable executable = (PreparedMetalExecutable) gpuSteps.getFirst().metadata().acceleratorExecutable();
        assertNotNull(executable.bridgeExecutable());
    }

    @Test
    void gpuMetalPartitionPrepareBuildsSingleAnchorStepForMatmulMulDivTanhChain() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{3, 2}, null, "b", DataType.FLOAT32);
        Tensor scale = new Tensor(new float[]{0.5f, 1.5f}, new int[]{2}, null, "scale", DataType.FLOAT32);
        Tensor denom = new Tensor(new float[]{2.0f, 4.0f}, new int[]{2}, null, "denom", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor mul = matmul.mul(scale);
        Tensor div = mul.div(denom);
        Tensor out = div.tanh();

        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(mul, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(div, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .toList();
        assertEquals(1, gpuSteps.size());
        PreparedMetalExecutable executable = (PreparedMetalExecutable) gpuSteps.getFirst().metadata().acceleratorExecutable();
        assertEquals(2, executable.plan().matMulSpec().postOps().stream().filter(postOp -> postOp.type().binary()).count());
    }

    @Test
    void gpuMetalPartitionPrepareBuildsSingleAnchorStepForMatmulSubTanhChain() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{3, 2}, null, "b", DataType.FLOAT32);
        Tensor shift = new Tensor(new float[]{0.5f, 1.5f}, new int[]{2}, null, "shift", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor sub = matmul.sub(shift);
        Tensor out = sub.tanh();

        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(sub, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .toList();
        assertEquals(1, gpuSteps.size());
        PreparedMetalExecutable executable = (PreparedMetalExecutable) gpuSteps.getFirst().metadata().acceleratorExecutable();
        assertEquals(1, executable.plan().matMulSpec().postOps().stream().filter(postOp -> postOp.type().binary()).count());
    }

    @Test
    void gpuMetalPartitionPrepareBuildsSingleAnchorStepForMatmulClampChain() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{3, 2}, null, "b", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor clampMin = matmul.clampMin(0.25);
        Tensor clampMax = clampMin.clampMax(5.0);
        Tensor out = clampMax.tanh();

        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(clampMin, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(clampMax, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .toList();
        assertEquals(1, gpuSteps.size());
        PreparedMetalExecutable executable = (PreparedMetalExecutable) gpuSteps.getFirst().metadata().acceleratorExecutable();
        assertEquals(2, executable.plan().matMulSpec().postOps().stream().filter(postOp -> postOp.hasScalarValue()).count());
    }

    @Test
    void gpuMetalPartitionPrepareBuildsSingleAnchorStepForMatmulBiasAddThenGenericAddChain() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{3, 2}, null, "b", DataType.FLOAT32);
        Tensor bias = new Tensor(new float[]{0.5f, 1.5f}, new int[]{2}, null, "bias", DataType.FLOAT32);
        Tensor residual = new Tensor(new float[]{0.25f, 0.75f}, new int[]{2}, null, "residual", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor biased = matmul.add(bias);
        Tensor added = biased.add(residual);
        Tensor out = added.tanh();

        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(biased, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(added, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .toList();
        assertEquals(1, gpuSteps.size());
        PreparedMetalExecutable executable = (PreparedMetalExecutable) gpuSteps.getFirst().metadata().acceleratorExecutable();
        assertEquals(1, executable.plan().matMulSpec().postOps().stream().filter(postOp -> postOp.type().binary()).count());
    }

    @Test
    void gpuMetalPartitionPrepareBuildsSingleAnchorStepForBranchMergeDag() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{3, 2}, null, "b", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor relu = matmul.relu();
        Tensor abs = matmul.abs();
        Tensor add = relu.add(abs);
        Tensor out = add.tanh();

        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(relu, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(abs, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(add, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .toList();
        assertEquals(1, gpuSteps.size());
        PreparedMetalExecutable executable = (PreparedMetalExecutable) gpuSteps.getFirst().metadata().acceleratorExecutable();
        assertEquals(5, executable.plan().lowering().dagSpec().nodes().size());
        assertEquals(5, executable.plan().subgraph().orderedNodeIds().size());
    }

    @Test
    void gpuMetalPartitionPrepareBuildsSingleAnchorStepForMultiMergeDag() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{3, 2}, null, "b", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor relu = matmul.relu();
        Tensor abs = matmul.abs();
        Tensor neg = matmul.neg();
        Tensor add1 = relu.add(abs);
        Tensor add2 = add1.add(neg);
        Tensor out = add2.tanh();

        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(relu, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(abs, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(neg, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(add1, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(add2, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .toList();
        assertEquals(1, gpuSteps.size());
        PreparedMetalExecutable executable = (PreparedMetalExecutable) gpuSteps.getFirst().metadata().acceleratorExecutable();
        assertEquals(7, executable.plan().lowering().dagSpec().nodes().size());
        assertTrue(execution.prepareTrace().backendSelection().selectedCount() >= 1);
    }

    @Test
    void gpuMetalPartitionSearchBudgetCanLimitAcceptedDagSize() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{3, 2}, null, "b", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor relu = matmul.relu();
        Tensor abs = matmul.abs();
        Tensor add = relu.add(abs);
        Tensor out = add.tanh();

        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(relu, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(abs, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(add, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        OptimizerConfig optimizer = OptimizerConfig.noOptimization().withPartition(
                new PartitionConfig(1, 4, 1000.0, 120.0, 450.0, 80.0, 60.0, 1.0)
        );
        PreparedExecution execution = CompiledGraph.compile(out, optimizer)
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .toList();
        assertEquals(1, gpuSteps.size());
        PreparedMetalExecutable executable = (PreparedMetalExecutable) gpuSteps.getFirst().metadata().acceleratorExecutable();
        assertEquals(1, executable.plan().lowering().dagSpec().nodes().size());
    }

    @Test
    void gpuMetalPartitionPrepareBuildsSingleAnchorStepForReshapeDag() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "b", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor reshape = matmul.reshape(1, 4);
        Tensor out = reshape.tanh();

        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(reshape, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .toList();
        assertEquals(1, gpuSteps.size());
        PreparedMetalExecutable executable = (PreparedMetalExecutable) gpuSteps.getFirst().metadata().acceleratorExecutable();
        assertEquals(3, executable.plan().lowering().dagSpec().nodes().size());
    }

    @Test
    void gpuMetalPartitionPrepareBuildsSingleAnchorStepForReshapeContiguousDag() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "b", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor reshape = matmul.reshape(1, 4);
        Tensor contiguous = reshape.contiguous();
        Tensor out = contiguous.neg();

        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(reshape, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(contiguous, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .toList();
        assertEquals(1, gpuSteps.size());
        PreparedMetalExecutable executable = (PreparedMetalExecutable) gpuSteps.getFirst().metadata().acceleratorExecutable();
        assertEquals(4, executable.plan().lowering().dagSpec().nodes().size());
    }

    @Test
    void gpuMetalPartitionPrepareBuildsSingleAnchorStepForPermuteDag() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "b", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor permute = matmul.permute(1, 0);
        Tensor out = permute.neg();

        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(permute, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .toList();
        assertEquals(1, gpuSteps.size());
        PreparedMetalExecutable executable = (PreparedMetalExecutable) gpuSteps.getFirst().metadata().acceleratorExecutable();
        assertEquals(3, executable.plan().lowering().dagSpec().nodes().size());
    }

    @Test
    void gpuMetalPartitionPrepareBuildsSingleAnchorStepForExpandSqueezeDag() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "b", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor reshape = matmul.reshape(1, 4);
        Tensor expand = reshape.expandDims(0);
        Tensor squeeze = expand.squeeze(0);
        Tensor out = squeeze.tanh();

        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(reshape, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(expand, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(squeeze, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .toList();
        assertEquals(1, gpuSteps.size());
        PreparedMetalExecutable executable = (PreparedMetalExecutable) gpuSteps.getFirst().metadata().acceleratorExecutable();
        assertEquals(5, executable.plan().lowering().dagSpec().nodes().size());
    }

    @Test
    void gpuMetalPartitionPrepareBuildsSingleAnchorStepForAttentionLikeRank4Slice() {
        Tensor q = new Tensor(new float[]{
                0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f,
                0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f
        }, new int[]{1, 2, 3, 2}, null, "q", DataType.FLOAT32);
        Tensor k = new Tensor(new float[]{
                0.2f, 0.1f, 0.4f, 0.3f, 0.6f, 0.5f,
                0.8f, 0.7f, 1.0f, 0.9f, 1.2f, 1.1f
        }, new int[]{1, 2, 3, 2}, null, "k", DataType.FLOAT32);
        Tensor kPermuted = k.permute(0, 1, 3, 2);
        Tensor scores = q.matmul(kPermuted);
        Tensor out = scores.mul(0.5);

        TensorInternalAccess.setBackend(kPermuted, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(scores, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .toList();
        assertEquals(1, gpuSteps.size());
        PreparedMetalExecutable executable = (PreparedMetalExecutable) gpuSteps.getFirst().metadata().acceleratorExecutable();
        assertEquals(3, executable.plan().lowering().dagSpec().nodes().size());
        assertEquals(2, executable.plan().lowering().dagSpec().externalInputs().size());
    }

    @Test
    void gpuMetalPartitionPrepareBuildsSingleAnchorStepForMaskedAttentionPreSoftmaxSlice() {
        Tensor q = new Tensor(new float[]{
                0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f,
                0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f
        }, new int[]{1, 2, 3, 2}, null, "q", DataType.FLOAT32);
        Tensor k = new Tensor(new float[]{
                0.2f, 0.1f, 0.4f, 0.3f, 0.6f, 0.5f,
                0.8f, 0.7f, 1.0f, 0.9f, 1.2f, 1.1f
        }, new int[]{1, 2, 3, 2}, null, "k", DataType.FLOAT32);
        Tensor mask = new Tensor(new byte[]{
                1, 1, 0,
                1, 0, 0,
                1, 1, 1,
                1, 0, 1,
                0, 0, 1,
                1, 1, 0
        }, new int[]{1, 2, 3, 3}, null, "mask", DataType.BOOL);
        Tensor fill = Tensor.scalar(-1.0e3, DataType.FLOAT32);
        Tensor kPermuted = k.permute(0, 1, 3, 2);
        Tensor matmul = q.matmul(kPermuted);
        Tensor scores = matmul.mul(0.5);
        Tensor out = Tensor.where(mask, scores, fill);

        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(kPermuted, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(scores, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .toList();
        assertEquals(1, gpuSteps.size());
        PreparedMetalExecutable executable = (PreparedMetalExecutable) gpuSteps.getFirst().metadata().acceleratorExecutable();
        assertEquals(4, executable.plan().lowering().dagSpec().nodes().size());
        assertEquals(4, executable.plan().lowering().dagSpec().externalInputs().size());
    }

    @Test
    void gpuMetalPartitionPrepareBuildsSingleAnchorStepForMaskedAttentionSoftmaxSlice() {
        Tensor q = new Tensor(new float[]{
                0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f,
                0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f
        }, new int[]{1, 2, 3, 2}, null, "q", DataType.FLOAT32);
        Tensor k = new Tensor(new float[]{
                0.2f, 0.1f, 0.4f, 0.3f, 0.6f, 0.5f,
                0.8f, 0.7f, 1.0f, 0.9f, 1.2f, 1.1f
        }, new int[]{1, 2, 3, 2}, null, "k", DataType.FLOAT32);
        Tensor mask = new Tensor(new byte[]{
                1, 1, 0,
                1, 0, 0,
                1, 1, 1,
                1, 0, 1,
                0, 0, 1,
                1, 1, 0
        }, new int[]{1, 2, 3, 3}, null, "mask", DataType.BOOL);
        Tensor fill = Tensor.scalar(-1.0e3, DataType.FLOAT32);
        Tensor kPermuted = k.permute(0, 1, 3, 2);
        Tensor matmul = q.matmul(kPermuted);
        Tensor scores = matmul.mul(0.5);
        Tensor masked = Tensor.where(mask, scores, fill);
        Tensor out = masked.softmax(3);

        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(kPermuted, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(scores, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(masked, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .toList();
        assertEquals(1, gpuSteps.size());
        PreparedMetalExecutable executable = (PreparedMetalExecutable) gpuSteps.getFirst().metadata().acceleratorExecutable();
        assertEquals(5, executable.plan().lowering().dagSpec().nodes().size());
    }

    @Test
    void gpuMetalPartitionPrepareBuildsSingleAnchorStepForMaskedAttentionFullForwardSlice() {
        Tensor q = new Tensor(new float[]{
                0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f,
                0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f
        }, new int[]{1, 2, 3, 2}, null, "q", DataType.FLOAT32);
        Tensor k = new Tensor(new float[]{
                0.2f, 0.1f, 0.4f, 0.3f, 0.6f, 0.5f,
                0.8f, 0.7f, 1.0f, 0.9f, 1.2f, 1.1f
        }, new int[]{1, 2, 3, 2}, null, "k", DataType.FLOAT32);
        Tensor v = new Tensor(new float[]{
                0.15f, 0.25f, 0.35f, 0.45f, 0.55f, 0.65f,
                0.75f, 0.85f, 0.95f, 1.05f, 1.15f, 1.25f
        }, new int[]{1, 2, 3, 2}, null, "v", DataType.FLOAT32);
        Tensor mask = new Tensor(new byte[]{
                1, 1, 0,
                1, 0, 0,
                1, 1, 1,
                1, 0, 1,
                0, 0, 1,
                1, 1, 0
        }, new int[]{1, 2, 3, 3}, null, "mask", DataType.BOOL);
        Tensor fill = Tensor.scalar(-1.0e3, DataType.FLOAT32);
        Tensor kPermuted = k.permute(0, 1, 3, 2);
        Tensor matmul = q.matmul(kPermuted);
        Tensor scores = matmul.mul(0.5);
        Tensor masked = Tensor.where(mask, scores, fill);
        Tensor weights = masked.softmax(3);
        Tensor out = weights.matmul(v);

        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(kPermuted, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(scores, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(masked, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(weights, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .toList();
        assertEquals(1, gpuSteps.size());
        PreparedMetalExecutable executable = (PreparedMetalExecutable) gpuSteps.getFirst().metadata().acceleratorExecutable();
        assertEquals(6, executable.plan().lowering().dagSpec().nodes().size());
        assertTrue(executable.plan().lowering().dagSpec().nodes().stream()
                .anyMatch(node -> node.type() == backend.accelerator.dag.AcceleratorDagNodeType.WHERE));
        assertTrue(executable.plan().lowering().dagSpec().nodes().stream()
                .noneMatch(node -> node.type() == backend.accelerator.dag.AcceleratorDagNodeType.SDPA));
    }

    @Test
    void acceleratorSelectionCanDisableMetalOffloadAtPrepareTime() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{3, 2}, null, "b", DataType.FLOAT32);
        Tensor bias = new Tensor(new float[]{1f, -1f}, new int[]{2}, null, "bias", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor add = matmul.add(bias);
        Tensor out = add.tanh();

        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(add, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        RuntimeConfig runtime = RuntimeConfig.inferenceDefaults().withAccelerator(
                RuntimeConfig.inferenceDefaults().accelerator().withMetal(
                        RuntimeConfig.inferenceDefaults().accelerator().metal().withEnabled(false)
                )
        );
        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .prepare(runtime);

        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .toList();
        assertEquals(0, gpuSteps.size());
        assertEquals(0, execution.prepareTrace().backendSelection().selectedCount());
        assertEquals(1, execution.prepareTrace().backendSelection().rejectedCount());
        assertEquals("backend-disabled", execution.prepareTrace().backendSelection().decisions().getFirst().reason());
    }

    @Test
    void acceleratorSelectionCanRejectMetalCandidateByEstimatedWorkGate() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{3, 2}, null, "b", DataType.FLOAT32);
        Tensor bias = new Tensor(new float[]{1f, -1f}, new int[]{2}, null, "bias", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor add = matmul.add(bias);
        Tensor out = add.tanh();

        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(add, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        RuntimeConfig runtime = RuntimeConfig.inferenceDefaults().withAccelerator(
                RuntimeConfig.inferenceDefaults().accelerator().withMetal(
                        RuntimeConfig.inferenceDefaults().accelerator().metal().withMinimumEstimatedWork(Long.MAX_VALUE)
                )
        );
        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .prepare(runtime);

        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .toList();
        assertEquals(0, gpuSteps.size());
        assertEquals(0, execution.prepareTrace().backendSelection().selectedCount());
        assertEquals(1, execution.prepareTrace().backendSelection().rejectedCount());
        var decision = execution.prepareTrace().backendSelection().decisions().getFirst();
        assertEquals("estimated-work-below-minimum", decision.reason());
        assertTrue(decision.estimatedWork() > 0L);
        assertNotNull(decision.costSummary());
        assertEquals("PROFILE_DERIVED", decision.costSummary().preset());
    }

    @Test
    void backendSelectionUsesProfileDerivedCostPreset() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{5f, 6f, 7f, 8f}, new int[]{2, 2}, null, "b", DataType.FLOAT32);
        Tensor out = a.matmul(b).relu();

        PreparedExecution execution = CompiledGraph.compile(
                        out,
                        OptimizerConfig.inferenceDefaults().withOffload(OffloadConfig.acceleratorGreedy())
                )
                .prepare(RuntimeConfig.inferenceDefaults());

        var selectedDecision = execution.prepareTrace().backendSelection().decisions().stream()
                .filter(decision -> decision.selected()
                        && decision.selectedBackend() == ComputeBackend.GPU_METAL)
                .findFirst()
                .orElseThrow();

        assertNotNull(selectedDecision.costSummary());
        assertEquals("PROFILE_DERIVED", selectedDecision.costSummary().preset());
    }

    @Test
    void minimumWorkRejectionStillWinsOverProfileDerivedCost() {
        graph.optimizer.partition.PartitionPlan plan = new graph.optimizer.partition.PartitionPlan() {
            @Override
            public ComputeBackend backend() {
                return ComputeBackend.GPU_METAL;
            }

            @Override
            public int anchorNodeId() {
                return 1;
            }

            @Override
            public List<Integer> nodeIds() {
                return List.of(1, 2, 3);
            }

            @Override
            public List<Integer> externalInputNodeIds() {
                return List.of(0);
            }

            @Override
            public List<Integer> producedOutputNodeIds() {
                return List.of(3);
            }

            @Override
            public long estimatedWork() {
                return 10L;
            }
        };
        RuntimeConfig runtime = RuntimeConfig.inferenceDefaults().withAccelerator(new AcceleratorConfig(
                AcceleratorBackendConfig.disabled(),
                AcceleratorBackendConfig.disabled(),
                new AcceleratorBackendConfig(true, false, 1_000_000L)
        ));

        AcceleratorPlanCostModel.Decision decision = AcceleratorPlanCostModel.decide(plan, runtime);

        assertFalse(decision.accepted());
        assertEquals("estimated-work-below-minimum", decision.reason());
        assertNotNull(decision.costSummary());
        assertEquals("PROFILE_DERIVED", decision.costSummary().preset());
    }

    @Test
    void staticCostDoesNotSelectAcceleratorWhenCpuPathIsClearlyCompetitive() {
        graph.optimizer.partition.PartitionPlan tinyBoundaryHeavyPlan = new graph.optimizer.partition.PartitionPlan() {
            @Override
            public ComputeBackend backend() {
                return ComputeBackend.GPU_METAL;
            }

            @Override
            public int anchorNodeId() {
                return 10;
            }

            @Override
            public List<Integer> nodeIds() {
                return List.of(10);
            }

            @Override
            public List<Integer> externalInputNodeIds() {
                return List.of(1, 2, 3, 4, 5, 6, 7, 8);
            }

            @Override
            public List<Integer> producedOutputNodeIds() {
                return List.of(10, 11, 12, 13, 14, 15, 16, 17);
            }

            @Override
            public long estimatedWork() {
                return 1L;
            }
        };

        AcceleratorPlanCostModel.Decision decision = AcceleratorPlanCostModel.decide(
                tinyBoundaryHeavyPlan,
                RuntimeConfig.inferenceDefaults()
        );

        assertFalse(decision.accepted());
        assertEquals("rejected-materialization-cost", decision.reason());
        assertNotNull(decision.costSummary());

        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{4}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{5f, 6f, 7f, 8f}, new int[]{4}, null, "b", DataType.FLOAT32);
        Tensor out = a.add(b).relu();

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults())
                .prepare(RuntimeConfig.inferenceDefaults());

        assertTrue(execution.forwardSteps().stream()
                .anyMatch(step -> step.metadata().backend() == ComputeBackend.CPU));
    }

    @Test
    void partStagePropagatesGpuIntentFromOutputBackToMatmulChain() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{3, 2}, null, "b", DataType.FLOAT32);
        Tensor bias = new Tensor(new float[]{1f, -1f}, new int[]{2}, null, "bias", DataType.FLOAT32);
        Tensor out = a.matmul(b).add(bias).tanh();
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults())
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .toList();
        assertEquals(1, gpuSteps.size());
    }

    @Test
    void gpuMetalMatmulAddTanhCanExecuteThroughExplicitAppleShim() {
        String explicitLib = System.getProperty("synaptik.metal.mps.lib");
        assumeTrue(explicitLib != null && !explicitLib.isBlank());

        Tensor cpuA = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "cpuA", DataType.FLOAT32);
        Tensor cpuB = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{3, 2}, null, "cpuB", DataType.FLOAT32);
        Tensor cpuBias = new Tensor(new float[]{1f, -1f}, new int[]{2}, null, "cpuBias", DataType.FLOAT32);
        Tensor cpuOut = cpuA.matmul(cpuB).add(cpuBias).tanh();
        CompiledGraph.compile(cpuOut, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{3, 2}, null, "b", DataType.FLOAT32);
        Tensor bias = new Tensor(new float[]{1f, -1f}, new int[]{2}, null, "bias", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor add = matmul.add(bias);
        Tensor out = add.tanh();

        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(add, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .toList();
        assertEquals(1, gpuSteps.size());
        PreparedMetalExecutable executable = (PreparedMetalExecutable) gpuSteps.getFirst().metadata().acceleratorExecutable();
        assumeTrue(executable.bridgeContext().available());
        assumeTrue(executable.bridgeExecutable().available());

        execution.execute(ExecutionMode.FORWARD);

        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 1e-5);
    }

    @Test
    void gpuMetalMatmulNegAbsSqrtInvCanExecuteThroughExplicitAppleShim() {
        String explicitLib = System.getProperty("synaptik.metal.mps.lib");
        assumeTrue(explicitLib != null && !explicitLib.isBlank());

        Tensor cpuA = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "cpuA", DataType.FLOAT32);
        Tensor cpuB = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{3, 2}, null, "cpuB", DataType.FLOAT32);
        Tensor cpuOut = cpuA.matmul(cpuB).neg().abs().sqrt().inv();
        CompiledGraph.compile(cpuOut, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{3, 2}, null, "b", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor neg = matmul.neg();
        Tensor abs = neg.abs();
        Tensor sqrt = abs.sqrt();
        Tensor out = sqrt.inv();

        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(neg, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(abs, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(sqrt, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuStep = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .findFirst()
                .orElseThrow();
        PreparedMetalExecutable executable = (PreparedMetalExecutable) gpuStep.metadata().acceleratorExecutable();
        assumeTrue(executable.bridgeExecutable().available());

        execution.execute(ExecutionMode.FORWARD);

        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 1e-5);
    }

    @Test
    void gpuMetalMatmulMulDivTanhCanExecuteThroughExplicitAppleShim() {
        String explicitLib = System.getProperty("synaptik.metal.mps.lib");
        assumeTrue(explicitLib != null && !explicitLib.isBlank());

        Tensor cpuA = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "cpuA", DataType.FLOAT32);
        Tensor cpuB = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{3, 2}, null, "cpuB", DataType.FLOAT32);
        Tensor cpuScale = new Tensor(new float[]{0.5f, 1.5f}, new int[]{2}, null, "cpuScale", DataType.FLOAT32);
        Tensor cpuDenom = new Tensor(new float[]{2.0f, 4.0f}, new int[]{2}, null, "cpuDenom", DataType.FLOAT32);
        Tensor cpuOut = cpuA.matmul(cpuB).mul(cpuScale).div(cpuDenom).tanh();
        CompiledGraph.compile(cpuOut, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{3, 2}, null, "b", DataType.FLOAT32);
        Tensor scale = new Tensor(new float[]{0.5f, 1.5f}, new int[]{2}, null, "scale", DataType.FLOAT32);
        Tensor denom = new Tensor(new float[]{2.0f, 4.0f}, new int[]{2}, null, "denom", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor mul = matmul.mul(scale);
        Tensor div = mul.div(denom);
        Tensor out = div.tanh();

        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(mul, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(div, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuStep = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .findFirst()
                .orElseThrow();
        PreparedMetalExecutable executable = (PreparedMetalExecutable) gpuStep.metadata().acceleratorExecutable();
        assumeTrue(executable.bridgeExecutable().available());

        execution.execute(ExecutionMode.FORWARD);

        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 1e-5);
    }

    @Test
    void gpuMetalMatmulSubTanhCanExecuteThroughExplicitAppleShim() {
        String explicitLib = System.getProperty("synaptik.metal.mps.lib");
        assumeTrue(explicitLib != null && !explicitLib.isBlank());

        Tensor cpuA = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "cpuA", DataType.FLOAT32);
        Tensor cpuB = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{3, 2}, null, "cpuB", DataType.FLOAT32);
        Tensor cpuShift = new Tensor(new float[]{0.5f, 1.5f}, new int[]{2}, null, "cpuShift", DataType.FLOAT32);
        Tensor cpuOut = cpuA.matmul(cpuB).sub(cpuShift).tanh();
        CompiledGraph.compile(cpuOut, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{3, 2}, null, "b", DataType.FLOAT32);
        Tensor shift = new Tensor(new float[]{0.5f, 1.5f}, new int[]{2}, null, "shift", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor sub = matmul.sub(shift);
        Tensor out = sub.tanh();

        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(sub, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuStep = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .findFirst()
                .orElseThrow();
        PreparedMetalExecutable executable = (PreparedMetalExecutable) gpuStep.metadata().acceleratorExecutable();
        assumeTrue(executable.bridgeExecutable().available());

        execution.execute(ExecutionMode.FORWARD);

        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 1e-5);
    }

    @Test
    void gpuMetalMatmulClampTanhCanExecuteThroughExplicitAppleShim() {
        String explicitLib = System.getProperty("synaptik.metal.mps.lib");
        assumeTrue(explicitLib != null && !explicitLib.isBlank());

        Tensor cpuA = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "cpuA", DataType.FLOAT32);
        Tensor cpuB = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{3, 2}, null, "cpuB", DataType.FLOAT32);
        Tensor cpuOut = cpuA.matmul(cpuB).clampMin(0.25).clampMax(5.0).tanh();
        CompiledGraph.compile(cpuOut, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{3, 2}, null, "b", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor clampMin = matmul.clampMin(0.25);
        Tensor clampMax = clampMin.clampMax(5.0);
        Tensor out = clampMax.tanh();

        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(clampMin, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(clampMax, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuStep = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .findFirst()
                .orElseThrow();
        PreparedMetalExecutable executable = (PreparedMetalExecutable) gpuStep.metadata().acceleratorExecutable();
        assumeTrue(executable.bridgeExecutable().available());

        execution.execute(ExecutionMode.FORWARD);

        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 1e-5);
    }

    @Test
    void gpuMetalMatmulBiasAddThenGenericAddCanExecuteThroughExplicitAppleShim() {
        String explicitLib = System.getProperty("synaptik.metal.mps.lib");
        assumeTrue(explicitLib != null && !explicitLib.isBlank());

        Tensor cpuA = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "cpuA", DataType.FLOAT32);
        Tensor cpuB = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{3, 2}, null, "cpuB", DataType.FLOAT32);
        Tensor cpuBias = new Tensor(new float[]{0.5f, 1.5f}, new int[]{2}, null, "cpuBias", DataType.FLOAT32);
        Tensor cpuResidual = new Tensor(new float[]{0.25f, 0.75f}, new int[]{2}, null, "cpuResidual", DataType.FLOAT32);
        Tensor cpuOut = cpuA.matmul(cpuB).add(cpuBias).add(cpuResidual).tanh();
        CompiledGraph.compile(cpuOut, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{3, 2}, null, "b", DataType.FLOAT32);
        Tensor bias = new Tensor(new float[]{0.5f, 1.5f}, new int[]{2}, null, "bias", DataType.FLOAT32);
        Tensor residual = new Tensor(new float[]{0.25f, 0.75f}, new int[]{2}, null, "residual", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor biased = matmul.add(bias);
        Tensor added = biased.add(residual);
        Tensor out = added.tanh();

        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(biased, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(added, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuStep = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .findFirst()
                .orElseThrow();
        PreparedMetalExecutable executable = (PreparedMetalExecutable) gpuStep.metadata().acceleratorExecutable();
        assumeTrue(executable.bridgeExecutable().available());

        execution.execute(ExecutionMode.FORWARD);

        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 1e-5);
    }

    @Test
    void gpuMetalBranchMergeDagCanExecuteThroughExplicitAppleShim() {
        String explicitLib = System.getProperty("synaptik.metal.mps.lib");
        assumeTrue(explicitLib != null && !explicitLib.isBlank());

        Tensor cpuA = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "cpuA", DataType.FLOAT32);
        Tensor cpuB = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{3, 2}, null, "cpuB", DataType.FLOAT32);
        Tensor cpuMatmul = cpuA.matmul(cpuB);
        Tensor cpuOut = cpuMatmul.relu().add(cpuMatmul.abs()).tanh();
        CompiledGraph.compile(cpuOut, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{3, 2}, null, "b", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor relu = matmul.relu();
        Tensor abs = matmul.abs();
        Tensor add = relu.add(abs);
        Tensor out = add.tanh();

        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(relu, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(abs, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(add, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuStep = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .findFirst()
                .orElseThrow();
        PreparedMetalExecutable executable = (PreparedMetalExecutable) gpuStep.metadata().acceleratorExecutable();
        assumeTrue(executable.bridgeExecutable().available());

        execution.execute(ExecutionMode.FORWARD);

        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 1e-5);
    }

    @Test
    void gpuMetalMultiMergeDagCanExecuteThroughExplicitAppleShim() {
        String explicitLib = System.getProperty("synaptik.metal.mps.lib");
        assumeTrue(explicitLib != null && !explicitLib.isBlank());

        Tensor cpuA = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "cpuA", DataType.FLOAT32);
        Tensor cpuB = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{3, 2}, null, "cpuB", DataType.FLOAT32);
        Tensor cpuMatmul = cpuA.matmul(cpuB);
        Tensor cpuOut = cpuMatmul.relu().add(cpuMatmul.abs()).add(cpuMatmul.neg()).tanh();
        CompiledGraph.compile(cpuOut, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{3, 2}, null, "b", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor relu = matmul.relu();
        Tensor abs = matmul.abs();
        Tensor neg = matmul.neg();
        Tensor add1 = relu.add(abs);
        Tensor add2 = add1.add(neg);
        Tensor out = add2.tanh();

        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(relu, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(abs, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(neg, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(add1, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(add2, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuStep = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .findFirst()
                .orElseThrow();
        PreparedMetalExecutable executable = (PreparedMetalExecutable) gpuStep.metadata().acceleratorExecutable();
        assumeTrue(executable.bridgeExecutable().available());

        execution.execute(ExecutionMode.FORWARD);

        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 1e-5);
    }

    @Test
    void gpuMetalReshapeDagCanExecuteThroughExplicitAppleShim() {
        String explicitLib = System.getProperty("synaptik.metal.mps.lib");
        assumeTrue(explicitLib != null && !explicitLib.isBlank());

        Tensor cpuA = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "cpuA", DataType.FLOAT32);
        Tensor cpuB = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "cpuB", DataType.FLOAT32);
        Tensor cpuOut = cpuA.matmul(cpuB).reshape(1, 4).tanh();
        CompiledGraph.compile(cpuOut, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "b", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor reshape = matmul.reshape(1, 4);
        Tensor out = reshape.tanh();

        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(reshape, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuStep = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .findFirst()
                .orElseThrow();
        PreparedMetalExecutable executable = (PreparedMetalExecutable) gpuStep.metadata().acceleratorExecutable();
        assumeTrue(executable.bridgeExecutable().available());

        execution.execute(ExecutionMode.FORWARD);

        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 1e-5);
    }

    @Test
    void gpuMetalPermuteDagCanExecuteThroughExplicitAppleShim() {
        String explicitLib = System.getProperty("synaptik.metal.mps.lib");
        assumeTrue(explicitLib != null && !explicitLib.isBlank());

        Tensor cpuA = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "cpuA", DataType.FLOAT32);
        Tensor cpuB = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "cpuB", DataType.FLOAT32);
        Tensor cpuOut = cpuA.matmul(cpuB).permute(1, 0).neg();
        CompiledGraph.compile(cpuOut, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "b", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor permute = matmul.permute(1, 0);
        Tensor out = permute.neg();

        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(permute, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuStep = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .findFirst()
                .orElseThrow();
        PreparedMetalExecutable executable = (PreparedMetalExecutable) gpuStep.metadata().acceleratorExecutable();
        assumeTrue(executable.bridgeExecutable().available());

        execution.execute(ExecutionMode.FORWARD);

        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 1e-5);
    }

    @Test
    void gpuMetalExpandSqueezeDagCanExecuteThroughExplicitAppleShim() {
        String explicitLib = System.getProperty("synaptik.metal.mps.lib");
        assumeTrue(explicitLib != null && !explicitLib.isBlank());

        Tensor cpuA = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "cpuA", DataType.FLOAT32);
        Tensor cpuB = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "cpuB", DataType.FLOAT32);
        Tensor cpuOut = cpuA.matmul(cpuB).reshape(1, 4).expandDims(0).squeeze(0).tanh();
        CompiledGraph.compile(cpuOut, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "b", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor reshape = matmul.reshape(1, 4);
        Tensor expand = reshape.expandDims(0);
        Tensor squeeze = expand.squeeze(0);
        Tensor out = squeeze.tanh();

        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(reshape, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(expand, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(squeeze, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuStep = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .findFirst()
                .orElseThrow();
        PreparedMetalExecutable executable = (PreparedMetalExecutable) gpuStep.metadata().acceleratorExecutable();
        assumeTrue(executable.bridgeExecutable().available());

        execution.execute(ExecutionMode.FORWARD);

        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 1e-5);
    }

    @Test
    void gpuMetalAttentionLikeRank4SliceCanExecuteThroughExplicitAppleShim() {
        String explicitLib = System.getProperty("synaptik.metal.mps.lib");
        assumeTrue(explicitLib != null && !explicitLib.isBlank());

        Tensor cpuQ = new Tensor(new float[]{
                0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f,
                0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f
        }, new int[]{1, 2, 3, 2}, null, "cpuQ", DataType.FLOAT32);
        Tensor cpuK = new Tensor(new float[]{
                0.2f, 0.1f, 0.4f, 0.3f, 0.6f, 0.5f,
                0.8f, 0.7f, 1.0f, 0.9f, 1.2f, 1.1f
        }, new int[]{1, 2, 3, 2}, null, "cpuK", DataType.FLOAT32);
        Tensor cpuOut = cpuQ.matmul(cpuK.permute(0, 1, 3, 2)).mul(0.5);
        CompiledGraph.compile(cpuOut, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor q = new Tensor(new float[]{
                0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f,
                0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f
        }, new int[]{1, 2, 3, 2}, null, "q", DataType.FLOAT32);
        Tensor k = new Tensor(new float[]{
                0.2f, 0.1f, 0.4f, 0.3f, 0.6f, 0.5f,
                0.8f, 0.7f, 1.0f, 0.9f, 1.2f, 1.1f
        }, new int[]{1, 2, 3, 2}, null, "k", DataType.FLOAT32);
        Tensor kPermuted = k.permute(0, 1, 3, 2);
        Tensor scores = q.matmul(kPermuted);
        Tensor out = scores.mul(0.5);

        TensorInternalAccess.setBackend(kPermuted, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(scores, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuStep = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .findFirst()
                .orElseThrow();
        PreparedMetalExecutable executable = (PreparedMetalExecutable) gpuStep.metadata().acceleratorExecutable();
        assumeTrue(executable.bridgeExecutable().available());

        execution.execute(ExecutionMode.FORWARD);

        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 1e-5);
    }

    @Test
    void gpuMetalMaskedAttentionPreSoftmaxSliceCanExecuteThroughExplicitAppleShim() {
        String explicitLib = System.getProperty("synaptik.metal.mps.lib");
        assumeTrue(explicitLib != null && !explicitLib.isBlank());

        Tensor cpuQ = new Tensor(new float[]{
                0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f,
                0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f
        }, new int[]{1, 2, 3, 2}, null, "cpuQ", DataType.FLOAT32);
        Tensor cpuK = new Tensor(new float[]{
                0.2f, 0.1f, 0.4f, 0.3f, 0.6f, 0.5f,
                0.8f, 0.7f, 1.0f, 0.9f, 1.2f, 1.1f
        }, new int[]{1, 2, 3, 2}, null, "cpuK", DataType.FLOAT32);
        Tensor cpuMask = new Tensor(new byte[]{
                1, 1, 0,
                1, 0, 0,
                1, 1, 1,
                1, 0, 1,
                0, 0, 1,
                1, 1, 0
        }, new int[]{1, 2, 3, 3}, null, "cpuMask", DataType.BOOL);
        Tensor cpuFill = Tensor.scalar(-1.0e3, DataType.FLOAT32);
        Tensor cpuOut = Tensor.where(cpuMask, cpuQ.matmul(cpuK.permute(0, 1, 3, 2)).mul(0.5), cpuFill);
        CompiledGraph.compile(cpuOut, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor q = new Tensor(new float[]{
                0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f,
                0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f
        }, new int[]{1, 2, 3, 2}, null, "q", DataType.FLOAT32);
        Tensor k = new Tensor(new float[]{
                0.2f, 0.1f, 0.4f, 0.3f, 0.6f, 0.5f,
                0.8f, 0.7f, 1.0f, 0.9f, 1.2f, 1.1f
        }, new int[]{1, 2, 3, 2}, null, "k", DataType.FLOAT32);
        Tensor mask = new Tensor(new byte[]{
                1, 1, 0,
                1, 0, 0,
                1, 1, 1,
                1, 0, 1,
                0, 0, 1,
                1, 1, 0
        }, new int[]{1, 2, 3, 3}, null, "mask", DataType.BOOL);
        Tensor fill = Tensor.scalar(-1.0e3, DataType.FLOAT32);
        Tensor kPermuted = k.permute(0, 1, 3, 2);
        Tensor matmul = q.matmul(kPermuted);
        Tensor scores = matmul.mul(0.5);
        Tensor out = Tensor.where(mask, scores, fill);

        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(kPermuted, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(scores, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuStep = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .findFirst()
                .orElseThrow();
        PreparedMetalExecutable executable = (PreparedMetalExecutable) gpuStep.metadata().acceleratorExecutable();
        assumeTrue(executable.bridgeExecutable().available());

        execution.execute(ExecutionMode.FORWARD);

        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 1e-5);
    }

    @Test
    void gpuMetalMaskedAttentionSoftmaxSliceCanExecuteThroughExplicitAppleShim() {
        String explicitLib = System.getProperty("synaptik.metal.mps.lib");
        assumeTrue(explicitLib != null && !explicitLib.isBlank());

        Tensor cpuQ = new Tensor(new float[]{
                0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f,
                0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f
        }, new int[]{1, 2, 3, 2}, null, "cpuQ", DataType.FLOAT32);
        Tensor cpuK = new Tensor(new float[]{
                0.2f, 0.1f, 0.4f, 0.3f, 0.6f, 0.5f,
                0.8f, 0.7f, 1.0f, 0.9f, 1.2f, 1.1f
        }, new int[]{1, 2, 3, 2}, null, "cpuK", DataType.FLOAT32);
        Tensor cpuMask = new Tensor(new byte[]{
                1, 1, 0,
                1, 0, 0,
                1, 1, 1,
                1, 0, 1,
                0, 0, 1,
                1, 1, 0
        }, new int[]{1, 2, 3, 3}, null, "cpuMask", DataType.BOOL);
        Tensor cpuFill = Tensor.scalar(-1.0e3, DataType.FLOAT32);
        Tensor cpuOut = Tensor.where(cpuMask, cpuQ.matmul(cpuK.permute(0, 1, 3, 2)).mul(0.5), cpuFill).softmax(3);
        CompiledGraph.compile(cpuOut, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor q = new Tensor(new float[]{
                0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f,
                0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f
        }, new int[]{1, 2, 3, 2}, null, "q", DataType.FLOAT32);
        Tensor k = new Tensor(new float[]{
                0.2f, 0.1f, 0.4f, 0.3f, 0.6f, 0.5f,
                0.8f, 0.7f, 1.0f, 0.9f, 1.2f, 1.1f
        }, new int[]{1, 2, 3, 2}, null, "k", DataType.FLOAT32);
        Tensor mask = new Tensor(new byte[]{
                1, 1, 0,
                1, 0, 0,
                1, 1, 1,
                1, 0, 1,
                0, 0, 1,
                1, 1, 0
        }, new int[]{1, 2, 3, 3}, null, "mask", DataType.BOOL);
        Tensor fill = Tensor.scalar(-1.0e3, DataType.FLOAT32);
        Tensor kPermuted = k.permute(0, 1, 3, 2);
        Tensor matmul = q.matmul(kPermuted);
        Tensor scores = matmul.mul(0.5);
        Tensor masked = Tensor.where(mask, scores, fill);
        Tensor out = masked.softmax(3);

        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(kPermuted, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(scores, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(masked, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuStep = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .findFirst()
                .orElseThrow();
        PreparedMetalExecutable executable = (PreparedMetalExecutable) gpuStep.metadata().acceleratorExecutable();
        assumeTrue(executable.bridgeExecutable().available());

        execution.execute(ExecutionMode.FORWARD);

        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 1e-5);
    }

    @Test
    void gpuMetalMaskedAttentionFullForwardSliceCanExecuteThroughExplicitAppleShim() {
        String explicitLib = System.getProperty("synaptik.metal.mps.lib");
        assumeTrue(explicitLib != null && !explicitLib.isBlank());

        Tensor cpuQ = new Tensor(new float[]{
                0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f,
                0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f
        }, new int[]{1, 2, 3, 2}, null, "cpuQ", DataType.FLOAT32);
        Tensor cpuK = new Tensor(new float[]{
                0.2f, 0.1f, 0.4f, 0.3f, 0.6f, 0.5f,
                0.8f, 0.7f, 1.0f, 0.9f, 1.2f, 1.1f
        }, new int[]{1, 2, 3, 2}, null, "cpuK", DataType.FLOAT32);
        Tensor cpuV = new Tensor(new float[]{
                0.15f, 0.25f, 0.35f, 0.45f, 0.55f, 0.65f,
                0.75f, 0.85f, 0.95f, 1.05f, 1.15f, 1.25f
        }, new int[]{1, 2, 3, 2}, null, "cpuV", DataType.FLOAT32);
        Tensor cpuMask = new Tensor(new byte[]{
                1, 1, 0,
                1, 0, 0,
                1, 1, 1,
                1, 0, 1,
                0, 0, 1,
                1, 1, 0
        }, new int[]{1, 2, 3, 3}, null, "cpuMask", DataType.BOOL);
        Tensor cpuFill = Tensor.scalar(-1.0e3, DataType.FLOAT32);
        Tensor cpuWeights = Tensor.where(cpuMask, cpuQ.matmul(cpuK.permute(0, 1, 3, 2)).mul(0.5), cpuFill).softmax(3);
        Tensor cpuOut = cpuWeights.matmul(cpuV);
        CompiledGraph.compile(cpuOut, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor q = new Tensor(new float[]{
                0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f,
                0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f
        }, new int[]{1, 2, 3, 2}, null, "q", DataType.FLOAT32);
        Tensor k = new Tensor(new float[]{
                0.2f, 0.1f, 0.4f, 0.3f, 0.6f, 0.5f,
                0.8f, 0.7f, 1.0f, 0.9f, 1.2f, 1.1f
        }, new int[]{1, 2, 3, 2}, null, "k", DataType.FLOAT32);
        Tensor v = new Tensor(new float[]{
                0.15f, 0.25f, 0.35f, 0.45f, 0.55f, 0.65f,
                0.75f, 0.85f, 0.95f, 1.05f, 1.15f, 1.25f
        }, new int[]{1, 2, 3, 2}, null, "v", DataType.FLOAT32);
        Tensor mask = new Tensor(new byte[]{
                1, 1, 0,
                1, 0, 0,
                1, 1, 1,
                1, 0, 1,
                0, 0, 1,
                1, 1, 0
        }, new int[]{1, 2, 3, 3}, null, "mask", DataType.BOOL);
        Tensor fill = Tensor.scalar(-1.0e3, DataType.FLOAT32);
        Tensor kPermuted = k.permute(0, 1, 3, 2);
        Tensor matmul = q.matmul(kPermuted);
        Tensor scores = matmul.mul(0.5);
        Tensor masked = Tensor.where(mask, scores, fill);
        Tensor weights = masked.softmax(3);
        Tensor out = weights.matmul(v);

        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(kPermuted, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(scores, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(masked, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(weights, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuStep = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .findFirst()
                .orElseThrow();
        PreparedMetalExecutable executable = (PreparedMetalExecutable) gpuStep.metadata().acceleratorExecutable();
        assumeTrue(executable.bridgeExecutable().available());

        execution.execute(ExecutionMode.FORWARD);

        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 1e-5);
    }

    @Test
    void gpuMetalMatmulAddExpCanExecuteThroughExplicitAppleShim() {
        String explicitLib = System.getProperty("synaptik.metal.mps.lib");
        assumeTrue(explicitLib != null && !explicitLib.isBlank());

        Tensor cpuA = new Tensor(new float[]{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f}, new int[]{2, 3}, null, "cpuA", DataType.FLOAT32);
        Tensor cpuB = new Tensor(new float[]{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f}, new int[]{3, 2}, null, "cpuB", DataType.FLOAT32);
        Tensor cpuBias = new Tensor(new float[]{0.1f, 0.2f}, new int[]{2}, null, "cpuBias", DataType.FLOAT32);
        Tensor cpuOut = cpuA.matmul(cpuB).add(cpuBias).exp();
        CompiledGraph.compile(cpuOut, OptimizerConfig.noOptimization())
                .execute(RuntimeConfig.inferenceDefaults(), ExecutionMode.FORWARD);

        Tensor a = new Tensor(new float[]{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f}, new int[]{2, 3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f}, new int[]{3, 2}, null, "b", DataType.FLOAT32);
        Tensor bias = new Tensor(new float[]{0.1f, 0.2f}, new int[]{2}, null, "bias", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor add = matmul.add(bias);
        Tensor out = add.exp();

        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(add, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.noOptimization())
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .toList();
        assertEquals(1, gpuSteps.size());
        PreparedMetalExecutable executable = (PreparedMetalExecutable) gpuSteps.getFirst().metadata().acceleratorExecutable();
        assumeTrue(executable.bridgeContext().available());
        assumeTrue(executable.bridgeExecutable().available());

        execution.execute(ExecutionMode.FORWARD);

        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 1e-5);
    }

    @Test
    void bfloat16LinearToLogSoftmaxPreparesBfloat16ReductionPath() {
        Tensor input = new Tensor(new double[32 * 64], new int[]{32, 64}, null, "input", DataType.BFLOAT16);
        Tensor weight = new Tensor(new double[64 * 96], new int[]{64, 96}, null, "weight", DataType.BFLOAT16);
        Tensor bias = new Tensor(new double[96], new int[]{96}, null, "bias", DataType.BFLOAT16);
        Tensor out = input.linear(weight, bias).logSoftmax(1);

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults())
                .prepare(bfloat16BlasRuntime());

        var linearStep = execution.forwardSteps().stream()
                .filter(step -> step.node().getOperation() != null && step.node().getOperation().opType() == Operation.OpType.LINEAR)
                .findFirst()
                .orElseThrow();
        var logSoftmaxStep = execution.forwardSteps().stream()
                .filter(step -> step.node().getOperation() != null && step.node().getOperation().opType() == Operation.OpType.LOG_SOFTMAX)
                .findFirst()
                .orElseThrow();

        assertTrue(linearStep.metadata().cpuPlan().publishFloatContinuation());
        assertEquals("BFLOAT16", logSoftmaxStep.metadata().cpuPlan().computeContract().storageType().name());
        assertEquals("F32", logSoftmaxStep.metadata().cpuPlan().computeContract().computeType().name());
        assertEquals("CPU_REDUCTION", logSoftmaxStep.metadata().cpuPlan().computeContract().backend().name());
    }

    @Test
    void bfloat16SoftmaxToMeanKeepsFloatContinuationInInference() {
        Tensor input = new Tensor(new double[32 * 64], new int[]{32, 64}, null, "input", DataType.BFLOAT16);
        Tensor weight = new Tensor(new double[64 * 96], new int[]{64, 96}, null, "weight", DataType.BFLOAT16);
        Tensor bias = new Tensor(new double[96], new int[]{96}, null, "bias", DataType.BFLOAT16);
        Tensor out = input.linear(weight, bias).softmax(1).mean(1);

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults())
                .prepare(bfloat16BlasRuntime());

        var linearStep = execution.forwardSteps().stream()
                .filter(step -> step.node().getOperation() != null && step.node().getOperation().opType() == Operation.OpType.LINEAR)
                .findFirst()
                .orElseThrow();
        var softmaxStep = execution.forwardSteps().stream()
                .filter(step -> step.node().getOperation() != null && step.node().getOperation().opType() == Operation.OpType.SOFTMAX)
                .findFirst()
                .orElseThrow();

        assertTrue(linearStep.metadata().cpuPlan().publishFloatContinuation());
        assertTrue(softmaxStep.metadata().cpuPlan().publishFloatContinuation());
        assertEquals("BFLOAT16", softmaxStep.metadata().cpuPlan().computeContract().storageType().name());
        assertEquals("F32", softmaxStep.metadata().cpuPlan().computeContract().computeType().name());
        assertEquals("CPU_REDUCTION", softmaxStep.metadata().cpuPlan().computeContract().backend().name());
    }

    @Test
    void bfloat16LayerNormToMeanKeepsFloatContinuationInInference() {
        Tensor input = new Tensor(new double[32 * 64], new int[]{32, 64}, null, "input", DataType.BFLOAT16);
        Tensor gamma = new Tensor(new double[64], new int[]{64}, null, "gamma", DataType.BFLOAT16);
        Tensor beta = new Tensor(new double[64], new int[]{64}, null, "beta", DataType.BFLOAT16);
        Tensor out = input.layerNorm(gamma, beta, 1e-5).mean(1);

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults())
                .prepare(bfloat16BlasRuntime());

        var layerNormStep = execution.forwardSteps().stream()
                .filter(step -> step.node().getOperation() != null && step.node().getOperation().opType() == Operation.OpType.LAYER_NORM)
                .findFirst()
                .orElseThrow();

        assertTrue(layerNormStep.metadata().cpuPlan().publishFloatContinuation());
        assertEquals("BFLOAT16", layerNormStep.metadata().cpuPlan().computeContract().storageType().name());
        assertEquals("F32", layerNormStep.metadata().cpuPlan().computeContract().computeType().name());
    }

    @Test
    void bfloat16RmsNormToMeanKeepsFloatContinuationInInference() {
        Tensor input = new Tensor(new double[32 * 64], new int[]{32, 64}, null, "input", DataType.BFLOAT16);
        Tensor gamma = new Tensor(new double[64], new int[]{64}, null, "gamma", DataType.BFLOAT16);
        Tensor out = input.rmsNorm(gamma, 1e-5).mean(1);

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults())
                .prepare(bfloat16BlasRuntime());

        var rmsNormStep = execution.forwardSteps().stream()
                .filter(step -> step.node().getOperation() != null && step.node().getOperation().opType() == Operation.OpType.RMS_NORM)
                .findFirst()
                .orElseThrow();

        assertTrue(rmsNormStep.metadata().cpuPlan().publishFloatContinuation());
        assertEquals("BFLOAT16", rmsNormStep.metadata().cpuPlan().computeContract().storageType().name());
        assertEquals("F32", rmsNormStep.metadata().cpuPlan().computeContract().computeType().name());
    }

    @Test
    void bfloat16LogSoftmaxToNllLossKeepsFloatContinuationInInference() {
        Tensor logits = new Tensor(new double[16 * 8], new int[]{16, 8}, null, "logits", DataType.BFLOAT16);
        Tensor targets = new Tensor(new double[16 * 8], new int[]{16, 8}, null, "targets", DataType.BFLOAT16);
        Tensor out = logits.logSoftmax(1).nllLoss(targets, 1);

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults())
                .prepare(bfloat16BlasRuntime());

        var logSoftmaxStep = execution.forwardSteps().stream()
                .filter(step -> step.node().getOperation() != null && step.node().getOperation().opType() == Operation.OpType.LOG_SOFTMAX)
                .findFirst()
                .orElseThrow();

        assertTrue(logSoftmaxStep.metadata().cpuPlan().publishFloatContinuation());
    }

    @Test
    void bfloat16MatmulToCrossEntropyLossPublishesFloatContinuationInInference() {
        Tensor a = new Tensor(new double[32 * 64], new int[]{32, 64}, null, "a", DataType.BFLOAT16);
        Tensor b = new Tensor(new double[64 * 96], new int[]{64, 96}, null, "b", DataType.BFLOAT16);
        Tensor targets = new Tensor(new double[32 * 96], new int[]{32, 96}, null, "targets", DataType.BFLOAT16);
        Tensor out = a.matmul(b).crossEntropyLoss(targets, 1);

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.inferenceDefaults())
                .prepare(bfloat16BlasRuntime());

        var matmulStep = execution.forwardSteps().stream()
                .filter(step -> step.node().getOperation() != null && step.node().getOperation().opType() == Operation.OpType.MATMUL)
                .findFirst()
                .orElseThrow();

        assertTrue(matmulStep.metadata().cpuPlan().publishFloatContinuation());
    }

    @Test
    void bfloat16SoftmaxGradPublishesFloatContinuationInTrainingUnaryChain() {
        Tensor input = new Tensor(new double[16 * 8], new int[]{16, 8}, null, "input", DataType.BFLOAT16);
        input.setRequiresGrad(true);
        Tensor out = input.exp().softmax(1).sum();

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.trainingDefaults())
                .prepare(bfloat16BlasRuntime());

        var softmaxGradStep = execution.backwardSteps().stream()
                .filter(step -> step.node().getOperation() != null && step.node().getOperation().opType() == Operation.OpType.SOFTMAX_GRAD)
                .findFirst()
                .orElseThrow();

        assertEquals("BFLOAT16", softmaxGradStep.metadata().cpuPlan().computeContract().storageType().name());
        assertEquals("F32", softmaxGradStep.metadata().cpuPlan().computeContract().computeType().name());
        assertEquals("CPU_REDUCTION", softmaxGradStep.metadata().cpuPlan().computeContract().backend().name());
        assertTrue(softmaxGradStep.metadata().cpuPlan().publishFloatContinuation());
    }

    @Test
    void bfloat16LogSoftmaxGradPublishesFloatContinuationInTrainingUnaryChain() {
        Tensor input = new Tensor(new double[16 * 8], new int[]{16, 8}, null, "input", DataType.BFLOAT16);
        input.setRequiresGrad(true);
        Tensor out = input.exp().logSoftmax(1).sum();

        PreparedExecution execution = CompiledGraph.compile(out, OptimizerConfig.trainingDefaults())
                .prepare(bfloat16BlasRuntime());

        var logSoftmaxGradStep = execution.backwardSteps().stream()
                .filter(step -> step.node().getOperation() != null && step.node().getOperation().opType() == Operation.OpType.LOG_SOFTMAX_GRAD)
                .findFirst()
                .orElseThrow();

        assertEquals("BFLOAT16", logSoftmaxGradStep.metadata().cpuPlan().computeContract().storageType().name());
        assertEquals("F32", logSoftmaxGradStep.metadata().cpuPlan().computeContract().computeType().name());
        assertEquals("CPU_REDUCTION", logSoftmaxGradStep.metadata().cpuPlan().computeContract().backend().name());
        assertTrue(logSoftmaxGradStep.metadata().cpuPlan().publishFloatContinuation());
    }

    private static RuntimeConfig bfloat16BlasRuntime() {
        return new RuntimeConfig(
                KernelTuningConfig.defaultsInference(),
                config.runtime.ApproximationConfig.defaults(),
                new config.runtime.BlasConfig(
                        BlasProvider.OPENBLAS_FFM,
                        1L,
                        false,
                        100.0d,
                        false,
                        1
                )
        );
    }

    private static KernelTuningConfig kernelWithVectorMin(int vectorMinSize) {
        return kernelWithVectorMinAndFusedAsmWidth(vectorMinSize, KernelTuningConfig.defaultsInference().cpu().fusedCheapContiguousAsmVectorWidth());
    }

    private static RuntimeConfig runtimeWithFusedAsmWidth(int fusedAsmWidth) {
        return new RuntimeConfig(
                kernelWithVectorMinAndFusedAsmWidth(1, fusedAsmWidth),
                config.runtime.ApproximationConfig.defaults(),
                config.runtime.BlasConfig.disabled(),
                new FusedExecutionPolicy(FusedPrimaryBackend.ASM, true)
        );
    }

    private static RuntimeConfig runtimeWithRequiredAcceleratorBuffer(ComputeBackend backend) {
        RuntimeConfig defaults = RuntimeConfig.inferenceDefaults();
        AcceleratorBackendConfig required = defaults.accelerator().forBackend(backend).withBuffer(
                new AcceleratorBufferConfig(AcceleratorBufferBindingMode.REQUIRE, true, Long.MAX_VALUE)
        );
        AcceleratorConfig accelerator = switch (backend) {
            case GPU_METAL -> defaults.accelerator().withMetal(required);
            case GPU_CUDA -> defaults.accelerator().withCuda(required);
            case GPU_OPENCL -> defaults.accelerator().withOpencl(required);
            case CPU -> defaults.accelerator();
        };
        return defaults.withAccelerator(accelerator);
    }

    private static KernelTuningConfig kernelWithVectorMinAndFusedAsmWidth(int vectorMinSize, int fusedAsmWidth) {
        var base = KernelTuningConfig.defaultsInference();
        var cpu = base.cpu();
        return new KernelTuningConfig(
                new config.backend.CpuKernelConfig(
                        cpu.loopUnrollFactor(),
                        cpu.matMulTileM(),
                        cpu.matMulTileN(),
                        cpu.matMulTileK(),
                        vectorMinSize,
                        vectorMinSize,
                        vectorMinSize,
                        vectorMinSize,
                        vectorMinSize,
                        cpu.parallelMinSize(),
                        cpu.parallelMinSize(),
                        cpu.fusedCheapParallelMinSize(),
                        cpu.fusedTranscendentalParallelMinSize(),
                        cpu.reductionParallelMinSize(),
                        cpu.contiguousMaterializeThreshold(),
                        cpu.lowCostTargetChunksPerWorker(),
                        cpu.mediumCostTargetChunksPerWorker(),
                        cpu.highCostTargetChunksPerWorker(),
                        cpu.minScalarChunkSize(),
                        cpu.minVectorChunkSize(),
                        cpu.minReductionChunkSize(),
                        cpu.commonPoolLowCostMaxWorkPerWorker(),
                        fusedAsmWidth,
                        fusedAsmWidth,
                        fusedAsmWidth,
                        fusedAsmWidth,
                        cpu.sumAccuracyMode(),
                        cpu.matMulParallelMinSize(),
                        cpu.attentionMatMulPolicy()
                ),
                base.cuda(),
                base.opencl()
        );
    }

    private static OptimizerConfig fuseOnlyInferenceConfig() {
        return new OptimizerConfig(
                java.util.List.of(config.optimizer.OptimizerStage.PART, config.optimizer.OptimizerStage.FUSE, config.optimizer.OptimizerStage.MEM),
                config.optimizer.RewriteConfig.defaults(),
                config.optimizer.CseConfig.strictDefaults(),
                config.optimizer.FuseConfig.inferenceDefaults(),
                config.optimizer.MemoryConfig.defaults()
        );
    }

    private static int nodeId(CompiledGraph compiled, Operation.OpType opType) {
        return compiled.compileArtifacts().compiledNodes().stream()
                .filter(node -> node.operation() != null && node.operation().opType() == opType)
                .map(graph.CompiledNode::id)
                .findFirst()
                .orElseThrow();
    }

    private static graph.CompiledNode compiledNode(CompiledGraph compiled, int nodeId) {
        return compiled.compileArtifacts().compiledNodes().stream()
                .filter(node -> node.id() == nodeId)
                .findFirst()
                .orElseThrow();
    }

    private static boolean hasSelectedAcceleratorDecisionFor(PreparedExecution execution, ComputeBackend backend, int nodeId) {
        return execution.prepareTrace().backendSelection().decisions().stream()
                .anyMatch(decision -> decision.selected()
                        && decision.selectedBackend() == backend
                        && decision.nodeIds().contains(nodeId));
    }

    private static void assertCpuPreparedStepAvailable(PreparedExecution execution, int nodeId) {
        assertTrue(execution.forwardSteps().stream()
                .anyMatch(step -> step.compiledNode().id() == nodeId && step.metadata().backend() == ComputeBackend.CPU));
    }

    private static void assertContainsAll(String actual, String... expectedSubstrings) {
        for (String expected : expectedSubstrings) {
            assertTrue(actual.contains(expected), () -> "Expected '" + actual + "' to contain '" + expected + "'");
        }
    }
}
