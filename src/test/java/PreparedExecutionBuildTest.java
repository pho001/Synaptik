import planning.descriptor.CompiledTensorDescriptorBuilder;
import planning.descriptor.CompiledTensorDescriptorIndex;
import backend.contract.ComputeBackend;
import backend.accelerator.buffer.AcceleratorBufferExecutionPath;
import backend.accelerator.exec.AcceleratorPreparedInputResolver;
import backend.accelerator.exec.ResolvedAcceleratorInputs;
import backend.accelerator.lowering.GpuCompoundPatternType;
import backend.accelerator.select.AcceleratorPlanCostModel;
import backend.cpu.plan.CpuNodeExecutionPlan;
import backend.cpu.plan.layout.StridedLayoutDecision;
import backend.cpu.plan.CpuLayoutPlan;
import backend.cpu.plan.CpuPreparedInput;
import backend.cpu1.kernels.matmul.Cpu1MatmulKernelId;
import backend.cpu1.prepare.Cpu1MatmulPostOp;
import backend.cpu1.prepare.Cpu1PreparedArtifact;
import backend.cpu1.prepare.Cpu1PreparedMatmulUnit;
import backend.cpu1.provider.matmul.Cpu1MatmulRoute;
import backend.cpu1.storage.Cpu1StorageKind;
import backend.cuda.lowering.CudaGpuBackendPartitionCapability;
import backend.metal.exec.PreparedMetalExecutable;
import backend.metal.lowering.MetalBackendPartitionCapability;
import backend.metal.lowering.MetalPartitionSupport;
import backend.cuda.exec.PreparedCudaExecutable;
import backend.runtime.ExecutionContext;
import runtime.contract.ExecutionMode;
import config.compile.CompileConfig;
import config.runtime.AcceleratorBackendConfig;
import config.runtime.AcceleratorBufferBindingMode;
import config.runtime.AcceleratorBufferConfig;
import config.runtime.AcceleratorConfig;
import config.runtime.RuntimeConfig;
import config.runtime.FusedExecutionPolicy;
import graph.CompiledGraph;
import graph.compile.CompiledNodeSnapshotter;
import graph.model.CompiledNode;
import graph.execution.plan.CompiledNodeExecutionMetadata;
import graph.execution.state.ExecutionState;
import graph.execution.PreparedExecution;
import graph.execution.PreparedExecutionStep;
import planning.partition.PartitionPlanningContext;
import operations.Operation;
import operations.index.gatherGrad;
import operations.index.takeAlongAxisGrad;
import backend.blas.BlasProvider;
import backend.blas.OpenBlasRuntime;
import config.backend.KernelTuningConfig;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.internal.TensorPrimitiveBuilder;
import tensor.layout.TensorRemap;
import tensor.options.AttentionOptions;
import tensor.options.Conv2dOptions;
import tensor.options.Pool2dOptions;
import planning.intent.BackendIntentPlan;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assumptions.assumeTrue;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

public class PreparedExecutionBuildTest {
    @Test
    void inferenceOnlyGraphBuildsForwardOnlyPreparedExecution() {
        Tensor a = new Tensor(new double[]{1.0, 2.0, 3.0, 4.0}, new int[]{4}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{0.5, 1.5, -2.0, 3.0}, new int[]{4}, null, "b", DataType.FLOAT64);
        Tensor out = a.add(b).mul(a).sigmoid();

        RuntimeConfig runtimeConfig = RuntimeConfig.inferenceDefaults();
        CompiledGraph compiledGraph = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline());
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
        CompiledGraph compiled = CompiledGraph.compile(out, CompileConfig.cpuOnlyBaseline());

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

        CompiledGraph compiled = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline());
        PreparedExecution execution = compiled.prepare(bfloat16BlasRuntime());

        assertThrows(UnsupportedOperationException.class, () -> execution.forwardSteps().clear());
        assertThrows(UnsupportedOperationException.class, () -> execution.backwardSteps().clear());
    }

    @Test
    void acceleratorLoweringArtifactsAreCompletedDuringCompile() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{5f, 6f, 7f, 8f}, new int[]{2, 2}, null, "b", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor out = matmul.relu();

        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();

        backendIntentPlan = backendIntentPlan.withBackend(matmul, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        CompileConfig partitionOnly = CompileConfig.inference()
                .withGraphOptimization(config.compile.GraphOptimizationConfig.noGraphOptimization());
        CompiledGraph compiled = CompiledGraph.compile(out, partitionOnly, backendIntentPlan);

        assertFalse(compiled.program().plannedPartitions().isEmpty());
        assertFalse(compiled.program().plannedRegions().isEmpty());
        assertNotNull(compiled.program().memoryPlan());
    }

    @Test
    void acceleratorOffloadCanSelectMetalForCpuFloat32GraphWithoutBackendIntent() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{5f, 6f, 7f, 8f}, new int[]{2, 2}, null, "b", DataType.FLOAT32);
        Tensor out = a.matmul(b).relu();

        CompileConfig optimizerConfig = CompileConfig.inference()
                .withBackendPlanning(config.compile.BackendPlanningConfig.autoAccelerator());
        CompiledGraph compiled = CompiledGraph.compile(out, optimizerConfig);

        assertTrue(compiled.program().compiledNodes().stream()
                .filter(node -> node.operation() != null)
                .allMatch(node -> node.backend() == ComputeBackend.CPU));
        assertTrue(compiled.program().plannedPartitions().stream()
                .anyMatch(partition -> partition.plan() != null
                        && partition.plan().backend() == ComputeBackend.GPU_METAL
                        && partition.nodeIds().size() >= 2));

        PreparedExecution execution = compiled.prepare(RuntimeConfig.inferenceDefaults());

        assertFalse(execution.prepareTrace().backendDiagnostics().isEmpty());
        var selectedDecision = execution.prepareTrace().backendSelection().decisions().stream()
                .filter(decision -> decision.selected()
                        && ComputeBackend.GPU_METAL.name().equals(decision.selectedBackend()))
                .findFirst()
                .orElseThrow();
        assertNotNull(selectedDecision.costSummary());
        assertEquals("PROFILE_DERIVED", selectedDecision.costSummary().preset());
        assertFalse(selectedDecision.costSummary().preset().isBlank());
        assertTrue(selectedDecision.costSummary().estimatedTransferBytes() >= 0L);
        assertTrue(execution.forwardSteps().stream()
                .anyMatch(step -> step.metadata().backend() == ComputeBackend.GPU_METAL
                        && testsupport.MetadataArtifacts.acceleratorExecutable(step.metadata()) instanceof PreparedMetalExecutable));
    }

    @Test
    void prepareTraceSelectedAcceleratorDecisionCarriesPlannerEvidence() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{5f, 6f, 7f, 8f}, new int[]{2, 2}, null, "b", DataType.FLOAT32);
        Tensor out = a.matmul(b).relu();

        CompileConfig optimizerConfig = CompileConfig.inference()
                .withBackendPlanning(config.compile.BackendPlanningConfig.autoAccelerator());
        PreparedExecution execution = CompiledGraph.compile(out, optimizerConfig)
                .prepare(RuntimeConfig.inferenceDefaults());

        var decision = execution.prepareTrace().backendSelection().decisions().stream()
                .filter(candidate -> candidate.selected()
                        && ComputeBackend.GPU_METAL.name().equals(candidate.selectedBackend()))
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
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(matmul, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        CompiledGraph compiled = CompiledGraph.compile(out, CompileConfig.inference(), backendIntentPlan);
        PreparedExecution execution = compiled.prepare(RuntimeConfig.inferenceDefaults());
        int matmulNodeId = nodeId(compiled, Operation.OpType.MATMUL);
        int logSoftmaxNodeId = nodeId(compiled, "logSoftmax");

        var selected = execution.prepareTrace().backendSelection().decisions().stream()
                .filter(decision -> decision.selected() && ComputeBackend.GPU_METAL.name().equals(decision.selectedBackend()))
                .filter(decision -> decision.nodeIds().contains(matmulNodeId) || decision.nodeIds().contains(logSoftmaxNodeId))
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
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(matmul, ComputeBackend.GPU_CUDA);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_CUDA);
        CompiledGraph compiled = CompiledGraph.compile(out, CompileConfig.inference(), backendIntentPlan);
        PreparedExecution execution = compiled.prepare(RuntimeConfig.inferenceDefaults());
        int matmulNodeId = nodeId(compiled, Operation.OpType.MATMUL);
        int logSoftmaxNodeId = nodeId(compiled, "logSoftmax");

        var selected = execution.prepareTrace().backendSelection().decisions().stream()
                .filter(decision -> decision.selected() && ComputeBackend.GPU_CUDA.name().equals(decision.selectedBackend()))
                .filter(decision -> decision.nodeIds().contains(matmulNodeId) || decision.nodeIds().contains(logSoftmaxNodeId))
                .findFirst()
                .orElseThrow();

        assertEquals("selected", selected.reason());
    }

    @Test
    void phaseNineteenSupportedMultiOpBufferPathDoesNotMaterializePreparedInputs() {
        Tensor input = new Tensor(new float[]{1f, -2f, 3f, -4f}, new int[]{4}, null, "phase19NativeBufferInput", DataType.FLOAT32);
        Tensor out = input.relu().exp();
        List<CompiledNode> nodes = CompiledNodeSnapshotter.snapshot(out.topologicalSort(), BackendIntentPlan.empty());
        Map<Integer, CompiledNodeExecutionMetadata> metadata = Map.of();
        ExecutionState state = ExecutionState.create(
                nodes,
                CompiledTensorDescriptorBuilder.build(nodes),
                metadata,
                nodes.getLast().id(),
                testsupport.PublicationPlans.forRoot(out, nodes, nodes.getLast().id())
        );
        ExecutionContext context = ExecutionContext.fromRuntimeConfig(
                RuntimeConfig.inferenceDefaults(),
                ExecutionMode.FORWARD,
                metadata,
                state
        );
        int inputNodeId = nodes.stream()
                .filter(node -> node.operation() == null)
                .map(CompiledNode::id)
                .findFirst()
                .orElseThrow();

        ResolvedAcceleratorInputs resolved = AcceleratorPreparedInputResolver.resolveForNativeBufferBinding(
                List.of(inputNodeId),
                context
        );

        assertEquals(List.of(inputNodeId), resolved.externalInputNodeIds());
        assertFalse(resolved.anyPreparedInputUsed());
        assertEquals(AcceleratorBufferExecutionPath.BUFFER_BINDING, AcceleratorBufferExecutionPath.BUFFER_BINDING);
        assertTrue(state.cpuMaterializationTraces().stream()
                .noneMatch(trace -> trace.reason() == runtime.contract.CpuMaterializationReason.ACCELERATOR_PREPARED_INPUT));
    }

    @Test
    void tensorArrayPreparedInputResolverDoesNotMaterializeUnmatchedInternalInputs() {
        Tensor external = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "resolverExternal", DataType.FLOAT32);
        Tensor internalSource = new Tensor(new float[]{5f, 6f, 7f, 8f}, new int[]{2, 2}, null, "resolverInternalSource", DataType.FLOAT32);
        Tensor internal = internalSource.relu();
        Tensor out = external.add(internal);
        List<CompiledNode> nodes = CompiledNodeSnapshotter.snapshot(out.topologicalSort(), BackendIntentPlan.empty());
        int externalNodeId = nodeId(nodes, "resolverExternal");
        int internalNodeId = nodeId(nodes, Operation.OpType.RELU);
        int consumerNodeId = nodeId(nodes, Operation.OpType.ADD);
        CompiledNode consumer = nodes.get(consumerNodeId);

        Tensor preparedExternal = new Tensor(external.getShape(), new ArrayList<>(), "preparedExternal", DataType.FLOAT32);
        Tensor preparedInternal = new Tensor(internal.getShape(), new ArrayList<>(), "preparedInternal", DataType.FLOAT32);
        CpuNodeExecutionPlan cpuPlan = new CpuNodeExecutionPlan(
                new CpuLayoutPlan(
                        StridedLayoutDecision.NONE,
                        DataType.FLOAT32,
                        0,
                        null,
                        null,
                        List.of(
                                new CpuPreparedInput(0, preparedExternal, TensorRemap.buildPlan(external, preparedExternal)),
                                new CpuPreparedInput(1, preparedInternal, TensorRemap.buildPlan(internal, preparedInternal))
                        )
                ),
                null,
                false,
                1,
                0,
                null,
                null,
                null,
                null,
                null
        );
        CompiledNodeExecutionMetadata metadata = testsupport.MetadataArtifacts.cpuMetadata(cpuPlan);
        Map<Integer, CompiledNodeExecutionMetadata> metadataIndex = Map.of(consumerNodeId, metadata);
        ExecutionState state = ExecutionState.create(
                nodes,
                CompiledTensorDescriptorBuilder.build(nodes),
                metadataIndex,
                nodes.getLast().id(),
                testsupport.PublicationPlans.forRoot(out, nodes, nodes.getLast().id())
        );
        ExecutionContext context = ExecutionContext.fromRuntimeConfig(
                RuntimeConfig.inferenceDefaults(),
                ExecutionMode.FORWARD,
                metadataIndex,
                state
        );

        ResolvedAcceleratorInputs resolved = AcceleratorPreparedInputResolver.resolve(
                List.of(new backend.accelerator.exec.PreparedAcceleratorExecutionSupport.CpuFallbackStep(consumer, metadata)),
                List.of(externalNodeId),
                context
        );

        assertTrue(resolved.anyPreparedInputUsed());
        assertEquals(List.of(externalNodeId), resolved.externalInputNodeIds());
        assertTrue(state.cpuMaterializationTraces().stream()
                .noneMatch(trace -> trace.nodeId() == internalNodeId
                        && trace.reason() == runtime.contract.CpuMaterializationReason.ACCELERATOR_PREPARED_INPUT));
    }

    @Test
    void metalCandidateRejectsOutputConsumedBeforeExecutionAnchor() {
        Tensor input = new Tensor(new float[]{1f, -2f, 3f, -4f}, new int[]{2, 2}, null, "earlyConsumerInput", DataType.FLOAT32);
        Tensor earlyProducer = input.relu();
        Tensor earlyCpuConsumer = earlyProducer.sum(1);
        Tensor laterProducer = input.exp();
        Tensor gpuMerge = earlyProducer.add(laterProducer);
        Tensor out = earlyCpuConsumer.add(gpuMerge.sum(1));
        List<CompiledNode> nodes = CompiledNodeSnapshotter.snapshot(out.topologicalSort(), BackendIntentPlan.empty());
        int earlyProducerNodeId = nodeId(nodes, Operation.OpType.RELU);
        int earlyConsumerNodeId = nodeId(nodes, Operation.OpType.SUM);
        int laterProducerNodeId = nodeId(nodes, Operation.OpType.EXP);
        int gpuMergeNodeId = nodeId(nodes, Operation.OpType.ADD);

        assertTrue(earlyProducerNodeId < earlyConsumerNodeId);
        assertTrue(earlyConsumerNodeId < gpuMergeNodeId);
        PartitionPlanningContext context = new PartitionPlanningContext(
                false,
                nodes,
                CompiledTensorDescriptorBuilder.build(nodes),
                consumerMap(nodes)
        );

        assertNull(new MetalBackendPartitionCapability().createCandidate(
                Set.of(earlyProducerNodeId, laterProducerNodeId, gpuMergeNodeId),
                context,
                Set.of()
        ));
    }

    @Test
    void metalSelectionAcceptsIndexTargetLossCandidate() {
        Tensor logits = new Tensor(new float[]{1f, 2f, 3f, 1f, 0f, -1f}, new int[]{2, 3}, null, "metalRejectedLossLogits", DataType.FLOAT32);
        Tensor targetIndices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "metalRejectedLossTargets", DataType.INT32);
        Tensor out = logits.crossEntropyLossFromIndices(targetIndices, 1);
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        CompiledGraph compiled = CompiledGraph.compile(out, CompileConfig.inference(), backendIntentPlan);
        PreparedExecution execution = compiled.prepare(RuntimeConfig.inferenceDefaults());
        int lossNodeId = nodeId(compiled, Operation.OpType.CROSS_ENTROPY_LOSS_INDICES);
        String plannerReason = MetalPartitionSupport.plannerUnsupportedReason(compiledNode(compiled, lossNodeId), planningContext(compiled));

        assertTrue(hasSelectedAcceleratorDecisionFor(execution, ComputeBackend.GPU_METAL, lossNodeId));
        assertTrue(plannerReason.isBlank(), plannerReason);
        assertAcceleratorPreparedStepAvailable(execution, lossNodeId);
    }

    @Test
    void metalSelectionAcceptsScatterAddWithSupportedProducerVisible() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "metalScatterInput", DataType.FLOAT32);
        Tensor weight = new Tensor(new float[]{
                1f, 0f, 0f,
                0f, 1f, 0f,
                0f, 0f, 1f
        }, new int[]{3, 3}, null, "metalScatterWeight", DataType.FLOAT32);
        Tensor indices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "metalScatterIndices", DataType.INT32);
        Tensor src = new Tensor(new float[]{0.5f, -0.25f}, new int[]{2}, null, "metalScatterSrc", DataType.FLOAT32);
        Tensor matmul = input.matmul(weight);
        Tensor out = matmul.scatterAdd(indices, src, 1);
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(matmul, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        CompiledGraph compiled = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline(), backendIntentPlan);
        PreparedExecution execution = compiled.prepare(RuntimeConfig.inferenceDefaults());
        int matmulNodeId = nodeId(compiled, Operation.OpType.MATMUL);
        int scatterNodeId = nodeId(compiled, Operation.OpType.SCATTER_ADD);
        String scatterReason = MetalPartitionSupport.plannerUnsupportedReason(compiledNode(compiled, scatterNodeId), planningContext(compiled));

        assertTrue(hasSelectedAcceleratorDecisionFor(execution, ComputeBackend.GPU_METAL, matmulNodeId));
        assertTrue(hasSelectedAcceleratorDecisionFor(execution, ComputeBackend.GPU_METAL, scatterNodeId));
        assertEquals("", scatterReason);
        assertAcceleratorPreparedStepAvailable(execution, scatterNodeId);
    }

    @Test
    void directCpuRejectsLegacyIndexGradientPrimitives() {
        Tensor gatherIndices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "metalGatherGradIndices", DataType.INT32);
        Tensor gatherOutGrad = new Tensor(new float[]{1f, 2f}, new int[]{2}, null, "metalGatherGradOut", DataType.FLOAT32);
        Tensor gatherGradOut = TensorPrimitiveBuilder.binary(
                gatherIndices,
                gatherOutGrad,
                new int[]{2, 3},
                new gatherGrad(1),
                "metalGatherGrad",
                DataType.FLOAT32
        );
        CompiledGraph gatherCompiled = CompiledGraph.compile(gatherGradOut, CompileConfig.noGraphOptimizationBaseline());
        IllegalStateException gatherError = assertThrows(
                IllegalStateException.class,
                () -> gatherCompiled.prepare(RuntimeConfig.inferenceDefaults())
        );
        assertTrue(gatherError.getMessage().contains("legacy backward op type GATHER_GRAD"));

        Tensor takeIndices = new Tensor(new int[]{2, 2, 0, 0}, new int[]{2, 2}, null, "metalTakeGradIndices", DataType.INT32);
        Tensor takeOutGrad = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "metalTakeGradOut", DataType.FLOAT32);
        Tensor takeGradOut = TensorPrimitiveBuilder.binary(
                takeIndices,
                takeOutGrad,
                new int[]{2, 3},
                new takeAlongAxisGrad(1),
                "metalTakeAlongAxisGrad",
                DataType.FLOAT32
        );
        CompiledGraph takeCompiled = CompiledGraph.compile(takeGradOut, CompileConfig.noGraphOptimizationBaseline());
        IllegalStateException takeError = assertThrows(
                IllegalStateException.class,
                () -> takeCompiled.prepare(RuntimeConfig.inferenceDefaults())
        );
        assertTrue(takeError.getMessage().contains("legacy backward op type TAKE_ALONG_AXIS_GRAD"));
    }

    @Test
    void cudaSelectionAcceptsSupportedReductionCandidateVisibly() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "cudaSupportedReductionInput", DataType.FLOAT32);
        Tensor out = input.sum(1);
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_CUDA);
        CompiledGraph compiled = CompiledGraph.compile(out, CompileConfig.inference(), backendIntentPlan);
        PreparedExecution execution = compiled.prepare(RuntimeConfig.inferenceDefaults());
        int reductionNodeId = nodeId(compiled, Operation.OpType.SUM);
        String plannerReason = CudaGpuBackendPartitionCapability.plannerUnsupportedReason(compiledNode(compiled, reductionNodeId), null);

        assertEquals("", plannerReason);
        assertTrue(hasSelectedAcceleratorDecisionFor(execution, ComputeBackend.GPU_CUDA, reductionNodeId));
        assertTrue(execution.forwardSteps().stream()
                .anyMatch(step -> step.compiledNode().id() == reductionNodeId
                        && step.metadata().backend() == ComputeBackend.GPU_CUDA
                        && testsupport.MetadataArtifacts.acceleratorExecutable(step.metadata()) instanceof PreparedCudaExecutable));
    }

    @Test
    void metalRequiredModeKeepsSupportedReductionOnAccelerator() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "metalRequiredReductionInput", DataType.FLOAT32);
        Tensor out = input.sum(1);
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        CompiledGraph compiled = CompiledGraph.compile(out, CompileConfig.inference(), backendIntentPlan);
        PreparedExecution execution = compiled.prepare(runtimeWithRequiredAcceleratorBuffer(ComputeBackend.GPU_METAL));
        int sumNodeId = nodeId(compiled, Operation.OpType.SUM);
        String plannerReason = MetalPartitionSupport.plannerUnsupportedReason(compiledNode(compiled, sumNodeId), null);

        assertEquals("", plannerReason);
        assertTrue(hasSelectedAcceleratorDecisionFor(execution, ComputeBackend.GPU_METAL, sumNodeId));
        assertTrue(execution.forwardSteps().stream()
                .anyMatch(step -> step.compiledNode().id() == sumNodeId
                        && step.metadata().backend() == ComputeBackend.GPU_METAL
                        && testsupport.MetadataArtifacts.acceleratorExecutable(step.metadata()) instanceof PreparedMetalExecutable));
    }

    @Test
    void metalRequiredModeKeepsScopedConv2dOnAccelerator() {
        Tensor input = new Tensor(new float[]{
                1f, 2f, 3f,
                4f, 5f, 6f,
                7f, 8f, 9f
        }, new int[]{1, 1, 3, 3}, null, "metalRequiredConvInput", DataType.FLOAT32);
        Tensor weight = new Tensor(new float[]{
                1f, 0f,
                0f, 1f
        }, new int[]{1, 1, 2, 2}, null, "metalRequiredConvWeight", DataType.FLOAT32);
        Tensor out = input.conv2d(weight, Conv2dOptions.defaults());
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        CompiledGraph compiled = CompiledGraph.compile(out, CompileConfig.inference(), backendIntentPlan);
        PreparedExecution execution = compiled.prepare(runtimeWithRequiredAcceleratorBuffer(ComputeBackend.GPU_METAL));
        int convNodeId = nodeId(compiled, Operation.OpType.CONV2D);
        String plannerReason = MetalPartitionSupport.plannerUnsupportedReason(compiledNode(compiled, convNodeId), planningContext(compiled));

        assertEquals("", plannerReason);
        assertTrue(hasSelectedAcceleratorDecisionFor(execution, ComputeBackend.GPU_METAL, convNodeId));
        PreparedMetalExecutable executable = execution.forwardSteps().stream()
                .filter(step -> step.compiledNode().id() == convNodeId
                        && step.metadata().backend() == ComputeBackend.GPU_METAL
                        && testsupport.MetadataArtifacts.acceleratorExecutable(step.metadata()) instanceof PreparedMetalExecutable)
                .map(step -> (PreparedMetalExecutable) testsupport.MetadataArtifacts.acceleratorExecutable(step.metadata()))
                .findFirst()
                .orElseThrow();
        assertTrue(executable.plan().lowering().dagSpec().nodes().stream()
                .anyMatch(node -> node.type() == backend.accelerator.dag.AcceleratorDagNodeType.CONV2D));
    }

    @Test
    void metalRequiredModeKeepsScopedPool2dOnAccelerator() {
        Tensor input = new Tensor(new float[]{
                1f, 2f, 3f, 4f,
                5f, 6f, 7f, 8f,
                9f, 10f, 11f, 12f,
                13f, 14f, 15f, 16f
        }, new int[]{1, 1, 4, 4}, null, "metalRequiredPoolInput", DataType.FLOAT32);
        Tensor out = input.maxPool2d(Pool2dOptions.square(2));
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        CompiledGraph compiled = CompiledGraph.compile(out, CompileConfig.inference(), backendIntentPlan);
        PreparedExecution execution = compiled.prepare(runtimeWithRequiredAcceleratorBuffer(ComputeBackend.GPU_METAL));
        int poolNodeId = nodeId(compiled, Operation.OpType.MAX_POOL2D);
        String plannerReason = MetalPartitionSupport.plannerUnsupportedReason(compiledNode(compiled, poolNodeId), planningContext(compiled));

        assertEquals("", plannerReason);
        assertTrue(hasSelectedAcceleratorDecisionFor(execution, ComputeBackend.GPU_METAL, poolNodeId));
        PreparedMetalExecutable executable = execution.forwardSteps().stream()
                .filter(step -> step.compiledNode().id() == poolNodeId
                        && step.metadata().backend() == ComputeBackend.GPU_METAL
                        && testsupport.MetadataArtifacts.acceleratorExecutable(step.metadata()) instanceof PreparedMetalExecutable)
                .map(step -> (PreparedMetalExecutable) testsupport.MetadataArtifacts.acceleratorExecutable(step.metadata()))
                .findFirst()
                .orElseThrow();
        assertTrue(executable.plan().lowering().dagSpec().nodes().stream()
                .anyMatch(node -> node.type() == backend.accelerator.dag.AcceleratorDagNodeType.MAX_POOL2D));
    }

    @Test
    void cudaRequiredModeSelectsPhaseTwentyFourNormalizationRegion() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "cudaRequiredPhase17NormInput", DataType.FLOAT32);
        Tensor gamma = new Tensor(new float[]{1f, 1f}, new int[]{2}, null, "cudaRequiredPhase17NormGamma", DataType.FLOAT32);
        Tensor beta = new Tensor(new float[]{0f, 0f}, new int[]{2}, null, "cudaRequiredPhase17NormBeta", DataType.FLOAT32);
        Tensor out = input.layerNorm(gamma, beta, 1.0e-5);
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_CUDA);
        CompiledGraph compiled = CompiledGraph.compile(out, CompileConfig.inference(), backendIntentPlan);
        PreparedExecution execution = compiled.prepare(runtimeWithRequiredAcceleratorBuffer(ComputeBackend.GPU_CUDA));
        int layerNormNodeId = nodeId(compiled, Operation.OpType.LAYER_NORM);
        String plannerReason = CudaGpuBackendPartitionCapability.plannerUnsupportedReason(
                compiledNode(compiled, layerNormNodeId),
                planningContext(compiled)
        );

        assertEquals("", plannerReason);
        assertTrue(hasSelectedAcceleratorDecisionFor(execution, ComputeBackend.GPU_CUDA, layerNormNodeId));
        assertTrue(execution.forwardSteps().stream()
                .anyMatch(step -> step.compiledNode().id() == layerNormNodeId
                        && step.metadata().backend() == ComputeBackend.GPU_CUDA
                        && testsupport.MetadataArtifacts.acceleratorExecutable(step.metadata()) instanceof PreparedCudaExecutable));
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
        CompiledGraph.compile(cpuOut, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        Tensor input = new Tensor(new float[]{0.25f, -0.5f, 1.25f, 2f, -1f, 0.75f}, new int[]{2, 3}, null, "phase17GpuLogSoftmaxInput", DataType.FLOAT32);
        Tensor weight = new Tensor(new float[]{
                1f, 0.5f, -0.25f,
                -0.75f, 1.25f, 0.5f,
                0.25f, -0.5f, 1.5f
        }, new int[]{3, 3}, null, "phase17GpuLogSoftmaxWeight", DataType.FLOAT32);
        Tensor matmul = input.matmul(weight);
        Tensor out = matmul.logSoftmax(1);
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(matmul, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        CompiledGraph compiled = CompiledGraph.compile(out, CompileConfig.inference(), backendIntentPlan);
        PreparedExecution execution = compiled.prepare(RuntimeConfig.inferenceDefaults());
        execution.execute(ExecutionMode.FORWARD);
        int logSoftmaxNodeId = nodeId(compiled, "logSoftmax");

        var manifest = execution.prepareTrace().backendSelection().decisions().stream()
                .filter(decision -> decision.selected() && ComputeBackend.GPU_METAL.name().equals(decision.selectedBackend()))
                .filter(decision -> decision.nodeIds().contains(logSoftmaxNodeId))
                .map(trace.prepare.BackendSelectionDecisionTrace::gpuLoweredRegionManifest)
                .filter(candidate -> candidate != null)
                .findFirst()
                .orElseThrow();
        String manifestText = tuning.benchmark.report.GpuLoweredRegionTraceRenderer.renderCompact(manifest);

        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 1.0e-5);
        assertFalse(manifestText.isBlank());
    }

    @Test
    void phaseSeventeenCrossEntropyIndexFallbackMatchesCpuAndReportsUnsupportedIndexSemantics() {
        Tensor cpuLogits = new Tensor(new float[]{1.5f, -0.25f, 0.5f, -1f, 2f, 0.25f}, new int[]{2, 3}, null, "phase17CpuLossLogits", DataType.FLOAT32);
        Tensor cpuTargets = new Tensor(new int[]{0, 1}, new int[]{2}, null, "phase17CpuLossTargets", DataType.INT32);
        Tensor cpuOut = cpuLogits.crossEntropyLossFromIndices(cpuTargets, 1);
        CompiledGraph.compile(cpuOut, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        Tensor logits = new Tensor(new float[]{1.5f, -0.25f, 0.5f, -1f, 2f, 0.25f}, new int[]{2, 3}, null, "phase17GpuLossLogits", DataType.FLOAT32);
        Tensor targets = new Tensor(new int[]{0, 1}, new int[]{2}, null, "phase17GpuLossTargets", DataType.INT32);
        Tensor out = logits.crossEntropyLossFromIndices(targets, 1);
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        CompiledGraph compiled = CompiledGraph.compile(out, CompileConfig.inference(), backendIntentPlan);
        PreparedExecution execution = compiled.prepare(RuntimeConfig.inferenceDefaults());
        execution.execute(ExecutionMode.FORWARD);
        int lossNodeId = nodeId(compiled, Operation.OpType.CROSS_ENTROPY_LOSS_INDICES);
        String plannerReason = MetalPartitionSupport.plannerUnsupportedReason(compiledNode(compiled, lossNodeId), planningContext(compiled));

        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 1.0e-5);
        assertTrue(plannerReason.isBlank(), plannerReason);
        assertTrue(hasSelectedAcceleratorDecisionFor(execution, ComputeBackend.GPU_METAL, lossNodeId));
    }

    @Test
    void phaseThirtyTwoTakeAlongAxisCanStayInsideMetalRegionAfterLogSoftmax() {
        Tensor input = new Tensor(new float[]{0.25f, -0.5f, 1.25f, 2f, -1f, 0.75f}, new int[]{2, 3}, null, "phase26GpuIndexInput", DataType.FLOAT32);
        Tensor weight = new Tensor(new float[]{
                1f, 0.5f, -0.25f,
                -0.75f, 1.25f, 0.5f,
                0.25f, -0.5f, 1.5f
        }, new int[]{3, 3}, null, "phase26GpuIndexWeight", DataType.FLOAT32);
        Tensor indices = new Tensor(new int[]{0, 1, 2, 0}, new int[]{2, 2}, null, "phase26GpuTakeIndices", DataType.INT32);
        Tensor matmul = input.matmul(weight);
        Tensor logProbs = matmul.logSoftmax(1);
        Tensor indexed = logProbs.takeAlongAxis(indices, 1);
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(matmul, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(logProbs, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(indexed, ComputeBackend.GPU_METAL);
        CompiledGraph compiled = CompiledGraph.compile(indexed, fuseOnlyInferenceConfig(), backendIntentPlan);
        PreparedExecution execution = compiled.prepare(RuntimeConfig.inferenceDefaults());
        int logSoftmaxNodeId = nodeId(compiled, "logSoftmax");
        int takeNodeId = nodeId(compiled, Operation.OpType.TAKE_ALONG_AXIS);
        String takeReason = MetalPartitionSupport.plannerUnsupportedReason(compiledNode(compiled, takeNodeId), planningContext(compiled));

        assertTrue(hasSelectedAcceleratorDecisionFor(execution, ComputeBackend.GPU_METAL, logSoftmaxNodeId),
                "legal LOG_SOFTMAX producer should stay selected on Metal");
        assertTrue(hasSelectedAcceleratorDecisionFor(execution, ComputeBackend.GPU_METAL, takeNodeId),
                "TAKE_ALONG_AXIS should stay selected once native INT32 index execution exists");
        assertTrue(execution.prepareTrace().backendSelection().decisions().stream()
                        .anyMatch(decision -> decision.selected()
                                && ComputeBackend.GPU_METAL.name().equals(decision.selectedBackend())
                                && decision.nodeIds().containsAll(List.of(logSoftmaxNodeId, takeNodeId))),
                "LOG_SOFTMAX producer and TAKE_ALONG_AXIS should be admitted into one Metal-owned region");
        assertEquals("", takeReason);
    }

    @Test
    void phaseTwentyFourLayerNormGpuRegionMatchesCpuAndReportsNormalizationDag() {
        Tensor cpuInput = new Tensor(new float[]{1f, 2f, 4f, 8f}, new int[]{2, 2}, null, "phase17CpuLayerNormInput", DataType.FLOAT32);
        Tensor cpuGamma = new Tensor(new float[]{1.25f, 0.75f}, new int[]{2}, null, "phase17CpuLayerNormGamma", DataType.FLOAT32);
        Tensor cpuBeta = new Tensor(new float[]{0.5f, -0.25f}, new int[]{2}, null, "phase17CpuLayerNormBeta", DataType.FLOAT32);
        Tensor cpuOut = cpuInput.layerNorm(cpuGamma, cpuBeta, 1.0e-5);
        CompiledGraph.compile(cpuOut, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        Tensor input = new Tensor(new float[]{1f, 2f, 4f, 8f}, new int[]{2, 2}, null, "phase17GpuLayerNormInput", DataType.FLOAT32);
        Tensor gamma = new Tensor(new float[]{1.25f, 0.75f}, new int[]{2}, null, "phase17GpuLayerNormGamma", DataType.FLOAT32);
        Tensor beta = new Tensor(new float[]{0.5f, -0.25f}, new int[]{2}, null, "phase17GpuLayerNormBeta", DataType.FLOAT32);
        Tensor out = input.layerNorm(gamma, beta, 1.0e-5);
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_CUDA);
        CompiledGraph compiled = CompiledGraph.compile(out, CompileConfig.inference(), backendIntentPlan);
        PreparedExecution execution = compiled.prepare(RuntimeConfig.inferenceDefaults());
        execution.execute(ExecutionMode.FORWARD);
        int layerNormNodeId = nodeId(compiled, Operation.OpType.LAYER_NORM);
        String plannerReason = CudaGpuBackendPartitionCapability.plannerUnsupportedReason(
                compiledNode(compiled, layerNormNodeId),
                planningContext(compiled)
        );

        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 1.0e-5);
        assertEquals("", plannerReason);
        assertTrue(hasSelectedAcceleratorDecisionFor(execution, ComputeBackend.GPU_CUDA, layerNormNodeId));
        PreparedCudaExecutable accelerator = (PreparedCudaExecutable) execution.forwardSteps().stream()
                .filter(step -> step.compiledNode().id() == layerNormNodeId)
                .map(step -> testsupport.MetadataArtifacts.acceleratorExecutable(step.metadata()))
                .filter(PreparedCudaExecutable.class::isInstance)
                .findFirst()
                .orElseThrow();
        assertEquals(GpuCompoundPatternType.NORMALIZATION, accelerator.compoundSummary().patternType());
        assertTrue(accelerator.dagSpec().nodes().size() > 5);
    }

    @Test
    void phaseTwentyFourRmsNormGpuRegionMatchesCpuAndReportsNormalizationDag() {
        Tensor cpuInput = new Tensor(new float[]{1f, 2f, 4f, 8f, 16f, 32f}, new int[]{2, 3}, null, "phase24CpuRmsNormInput", DataType.FLOAT32);
        Tensor cpuGamma = new Tensor(new float[]{1.25f, 0.75f, 1.5f}, new int[]{3}, null, "phase24CpuRmsNormGamma", DataType.FLOAT32);
        Tensor cpuOut = cpuInput.rmsNorm(cpuGamma, 1.0e-5);
        CompiledGraph.compile(cpuOut, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        Tensor input = new Tensor(new float[]{1f, 2f, 4f, 8f, 16f, 32f}, new int[]{2, 3}, null, "phase24GpuRmsNormInput", DataType.FLOAT32);
        Tensor gamma = new Tensor(new float[]{1.25f, 0.75f, 1.5f}, new int[]{3}, null, "phase24GpuRmsNormGamma", DataType.FLOAT32);
        Tensor out = input.rmsNorm(gamma, 1.0e-5);
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        CompiledGraph compiled = CompiledGraph.compile(out, CompileConfig.inference(), backendIntentPlan);
        PreparedExecution execution = compiled.prepare(RuntimeConfig.inferenceDefaults());
        execution.execute(ExecutionMode.FORWARD);
        int rmsNormNodeId = nodeId(compiled, Operation.OpType.RMS_NORM);
        String plannerReason = MetalPartitionSupport.plannerUnsupportedReason(
                compiledNode(compiled, rmsNormNodeId),
                planningContext(compiled)
        );

        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 1.0e-5);
        assertEquals("", plannerReason);
        assertTrue(hasSelectedAcceleratorDecisionFor(execution, ComputeBackend.GPU_METAL, rmsNormNodeId));
        PreparedMetalExecutable accelerator = (PreparedMetalExecutable) execution.forwardSteps().stream()
                .filter(step -> step.compiledNode().id() == rmsNormNodeId)
                .map(step -> testsupport.MetadataArtifacts.acceleratorExecutable(step.metadata()))
                .filter(PreparedMetalExecutable.class::isInstance)
                .findFirst()
                .orElseThrow();
        assertEquals(GpuCompoundPatternType.NORMALIZATION, accelerator.compoundSummary().patternType());
        assertTrue(accelerator.plan().lowering().dagSpec().nodes().size() > 5);
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
        CompiledGraph.compile(cpuOut, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "metalCompoundInput", DataType.FLOAT32);
        Tensor weight = new Tensor(new float[]{
                1f, 0f, 0f, 1f,
                0f, 1f, 1f, 0f,
                1f, 1f, 0f, 0f
        }, new int[]{3, 4}, null, "metalCompoundWeight", DataType.FLOAT32);
        Tensor bias = new Tensor(new float[]{0.5f, -0.5f, 1f, -1f}, new int[]{4}, null, "metalCompoundBias", DataType.FLOAT32);
        Tensor linear = input.linear(weight, bias);
        Tensor out = linear.relu();
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(linear, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        CompiledGraph compiled = CompiledGraph.compile(out, CompileConfig.inference(), backendIntentPlan);
        PreparedExecution execution = compiled.prepare(RuntimeConfig.inferenceDefaults());
        int linearNodeId = nodeId(compiled, Operation.OpType.LINEAR);
        int reluNodeId = nodeId(compiled, Operation.OpType.RELU);
        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .toList();

        assertEquals(1, gpuSteps.size());
        PreparedMetalExecutable executable = (PreparedMetalExecutable) testsupport.MetadataArtifacts.acceleratorExecutable(gpuSteps.getFirst().metadata());
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
        CompiledGraph.compile(cpuOut, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "cudaCompoundInput", DataType.FLOAT32);
        Tensor weight = new Tensor(new float[]{
                1f, 0f, 0f, 1f,
                0f, 1f, 1f, 0f,
                1f, 1f, 0f, 0f
        }, new int[]{3, 4}, null, "cudaCompoundWeight", DataType.FLOAT32);
        Tensor bias = new Tensor(new float[]{0.5f, -0.5f, 1f, -1f}, new int[]{4}, null, "cudaCompoundBias", DataType.FLOAT32);
        Tensor linear = input.linear(weight, bias);
        Tensor out = linear.relu();
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(linear, ComputeBackend.GPU_CUDA);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_CUDA);
        CompiledGraph compiled = CompiledGraph.compile(out, CompileConfig.inference(), backendIntentPlan);
        PreparedExecution execution = compiled.prepare(RuntimeConfig.inferenceDefaults());
        int linearNodeId = nodeId(compiled, Operation.OpType.LINEAR);
        int reluNodeId = nodeId(compiled, Operation.OpType.RELU);
        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_CUDA)
                .toList();

        assertEquals(1, gpuSteps.size());
        PreparedCudaExecutable executable = (PreparedCudaExecutable) testsupport.MetadataArtifacts.acceleratorExecutable(gpuSteps.getFirst().metadata());
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
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(linear, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        RuntimeConfig defaults = RuntimeConfig.inferenceDefaults();
        RuntimeConfig runtime = defaults.withAccelerator(defaults.accelerator().withMetal(
                defaults.accelerator().metal().withBuffer(
                        new AcceleratorBufferConfig(AcceleratorBufferBindingMode.REQUIRE, true, Long.MAX_VALUE)
                )
        ));
        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.inference(), backendIntentPlan)
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
        CompiledGraph.compile(cpuOut, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "metalEpilogueInput", DataType.FLOAT32);
        Tensor weight = new Tensor(new float[]{
                1f, 0f, 0f, 1f,
                0f, 1f, 1f, 0f,
                1f, 1f, 0f, 0f
        }, new int[]{3, 4}, null, "metalEpilogueWeight", DataType.FLOAT32);
        Tensor bias = new Tensor(new float[]{0.5f, -0.5f, 1f, -1f}, new int[]{4}, null, "metalEpilogueBias", DataType.FLOAT32);
        Tensor linear = input.linear(weight, bias);
        Tensor out = linear.relu();
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(linear, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        CompiledGraph compiled = CompiledGraph.compile(out, CompileConfig.inference(), backendIntentPlan);
        PreparedExecution execution = compiled.prepare(RuntimeConfig.inferenceDefaults());
        int linearNodeId = nodeId(compiled, Operation.OpType.LINEAR);
        int reluNodeId = nodeId(compiled, Operation.OpType.RELU);
        PreparedMetalExecutable executable = (PreparedMetalExecutable) execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .map(step -> testsupport.MetadataArtifacts.acceleratorExecutable(step.metadata()))
                .findFirst()
                .orElseThrow();

        assertTrue(executable.gpuLoweredRegionManifest().fusedSubpatterns().stream()
                .anyMatch(subpattern -> subpattern.patternType() == GpuCompoundPatternType.LINEAR_BIAS_ACTIVATION
                        && subpattern.detail().contains("epilogue")));
        var trace = execution.executeTraced(ExecutionMode.FORWARD);

        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 1.0e-5);
        assertFalse(trace.cpuMaterializations().stream()
                .anyMatch(entry -> (entry.nodeId() == linearNodeId || entry.nodeId() == reluNodeId)
                        && entry.reason() == runtime.contract.CpuMaterializationReason.CPU_CONSUMER));
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
        CompiledGraph.compile(cpuOut, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "cudaEpilogueInput", DataType.FLOAT32);
        Tensor weight = new Tensor(new float[]{
                1f, 0f, 0f, 1f,
                0f, 1f, 1f, 0f,
                1f, 1f, 0f, 0f
        }, new int[]{3, 4}, null, "cudaEpilogueWeight", DataType.FLOAT32);
        Tensor bias = new Tensor(new float[]{0.5f, -0.5f, 1f, -1f}, new int[]{4}, null, "cudaEpilogueBias", DataType.FLOAT32);
        Tensor linear = input.linear(weight, bias);
        Tensor out = linear.relu();
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(linear, ComputeBackend.GPU_CUDA);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_CUDA);
        CompiledGraph compiled = CompiledGraph.compile(out, CompileConfig.inference(), backendIntentPlan);
        PreparedExecution execution = compiled.prepare(RuntimeConfig.inferenceDefaults());
        int linearNodeId = nodeId(compiled, Operation.OpType.LINEAR);
        int reluNodeId = nodeId(compiled, Operation.OpType.RELU);
        PreparedCudaExecutable executable = (PreparedCudaExecutable) execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_CUDA)
                .map(step -> testsupport.MetadataArtifacts.acceleratorExecutable(step.metadata()))
                .findFirst()
                .orElseThrow();

        assertTrue(executable.gpuLoweredRegionManifest().fusedSubpatterns().stream()
                .anyMatch(subpattern -> subpattern.patternType() == GpuCompoundPatternType.LINEAR_BIAS_ACTIVATION
                        && subpattern.detail().contains("epilogue")));
        var trace = execution.executeTraced(ExecutionMode.FORWARD);

        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 1.0e-5);
        assertFalse(trace.cpuMaterializations().stream()
                .anyMatch(entry -> (entry.nodeId() == linearNodeId || entry.nodeId() == reluNodeId)
                        && entry.reason() == runtime.contract.CpuMaterializationReason.CPU_CONSUMER));
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
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(linear, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.inference(), backendIntentPlan)
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
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(add, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(relu, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        CompiledGraph compiled = CompiledGraph.compile(out, CompileConfig.inference(), backendIntentPlan);
        PreparedExecution execution = compiled.prepare(RuntimeConfig.inferenceDefaults());
        int addNodeId = nodeId(compiled, Operation.OpType.ADD);
        int reluNodeId = nodeId(compiled, Operation.OpType.RELU);
        int expNodeId = nodeId(compiled, Operation.OpType.EXP);
        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .toList();

        assertEquals(1, gpuSteps.size());
        PreparedMetalExecutable executable = (PreparedMetalExecutable) testsupport.MetadataArtifacts.acceleratorExecutable(gpuSteps.getFirst().metadata());
        assertEquals(GpuCompoundPatternType.ELEMENTWISE_CHAIN, executable.compoundSummary().patternType());
        assertTrue(executable.compoundSummary().supported());
        assertTrue(executable.compoundSummary().orderedNodeIds().containsAll(List.of(addNodeId, reluNodeId, expNodeId)));
        assertTrue(executable.compoundSummary().dagNodeTypes().containsAll(List.of("ADD", "RELU", "EXP")));
    }

    @Test
    void phaseNineteenMetalPreparedExecutableExposesMultiOpManifest() {
        Tensor a = new Tensor(new float[]{1f, -2f, 3f, -4f}, new int[]{4}, null, "phase19MetalManifestA", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{0.5f, 1f, -1f, 2f}, new int[]{4}, null, "phase19MetalManifestB", DataType.FLOAT32);
        Tensor add = a.add(b);
        Tensor relu = add.relu();
        Tensor out = relu.exp();
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(add, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(relu, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.inference(), backendIntentPlan)
                .prepare(RuntimeConfig.inferenceDefaults());
        PreparedMetalExecutable executable = (PreparedMetalExecutable) execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .map(step -> testsupport.MetadataArtifacts.acceleratorExecutable(step.metadata()))
                .findFirst()
                .orElseThrow();

        assertEquals(GpuCompoundPatternType.ELEMENTWISE_CHAIN, executable.compoundSummary().patternType());
        assertNotNull(executable.gpuLoweredRegionManifest());
        assertTrue(executable.gpuLoweredRegionManifest().selectedRegionLength() > 1);
        assertTrue(executable.gpuLoweredRegionManifest().loweredPrimitives().size() > 1);
        assertEquals(backend.accelerator.buffer.AcceleratorBufferExecutionPath.UNAVAILABLE,
                executable.lastAcceleratorBufferDecision().path());
    }

    @Test
    void gpuCudaElementwiseChainPublishesCompoundSummary() {
        Tensor a = new Tensor(new float[]{1f, -2f, 3f, -4f}, new int[]{4}, null, "cudaChainA", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{0.5f, 1f, -1f, 2f}, new int[]{4}, null, "cudaChainB", DataType.FLOAT32);
        Tensor add = a.add(b);
        Tensor relu = add.relu();
        Tensor out = relu.exp();
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(add, ComputeBackend.GPU_CUDA);
        backendIntentPlan = backendIntentPlan.withBackend(relu, ComputeBackend.GPU_CUDA);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_CUDA);
        CompiledGraph compiled = CompiledGraph.compile(out, CompileConfig.inference(), backendIntentPlan);
        PreparedExecution execution = compiled.prepare(RuntimeConfig.inferenceDefaults());
        int addNodeId = nodeId(compiled, Operation.OpType.ADD);
        int reluNodeId = nodeId(compiled, Operation.OpType.RELU);
        int expNodeId = nodeId(compiled, Operation.OpType.EXP);
        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_CUDA)
                .toList();

        assertEquals(1, gpuSteps.size());
        PreparedCudaExecutable executable = (PreparedCudaExecutable) testsupport.MetadataArtifacts.acceleratorExecutable(gpuSteps.getFirst().metadata());
        assertEquals(GpuCompoundPatternType.ELEMENTWISE_CHAIN, executable.compoundSummary().patternType());
        assertTrue(executable.compoundSummary().supported());
        assertTrue(executable.compoundSummary().orderedNodeIds().containsAll(List.of(addNodeId, reluNodeId, expNodeId)));
        assertTrue(executable.compoundSummary().dagNodeTypes().containsAll(List.of("ADD", "RELU", "EXP")));
    }

    @Test
    void phaseNineteenCudaPreparedExecutableExposesMultiOpManifest() {
        Tensor a = new Tensor(new float[]{1f, -2f, 3f, -4f}, new int[]{4}, null, "phase19CudaManifestA", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{0.5f, 1f, -1f, 2f}, new int[]{4}, null, "phase19CudaManifestB", DataType.FLOAT32);
        Tensor add = a.add(b);
        Tensor relu = add.relu();
        Tensor out = relu.exp();
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(add, ComputeBackend.GPU_CUDA);
        backendIntentPlan = backendIntentPlan.withBackend(relu, ComputeBackend.GPU_CUDA);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_CUDA);
        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.inference(), backendIntentPlan)
                .prepare(RuntimeConfig.inferenceDefaults());
        PreparedCudaExecutable executable = (PreparedCudaExecutable) execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_CUDA)
                .map(step -> testsupport.MetadataArtifacts.acceleratorExecutable(step.metadata()))
                .findFirst()
                .orElseThrow();

        assertEquals(GpuCompoundPatternType.ELEMENTWISE_CHAIN, executable.compoundSummary().patternType());
        assertNotNull(executable.gpuLoweredRegionManifest());
        assertTrue(executable.gpuLoweredRegionManifest().selectedRegionLength() > 1);
        assertTrue(executable.gpuLoweredRegionManifest().loweredPrimitives().size() > 1);
        assertEquals(backend.accelerator.buffer.AcceleratorBufferExecutionPath.UNAVAILABLE,
                executable.lastAcceleratorBufferDecision().path());
    }

    @Test
    void metalElementwiseFusionKeepsInteriorValuesDeviceOwned() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "metalInteriorA", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 0f, 0f, 1f, 1f, 1f}, new int[]{3, 2}, null, "metalInteriorB", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor relu = matmul.relu();
        Tensor exp = relu.exp();
        Tensor out = exp.log();
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(matmul, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(relu, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(exp, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        CompiledGraph compiled = CompiledGraph.compile(out, CompileConfig.inference(), backendIntentPlan);
        PreparedExecution execution = compiled.prepare(RuntimeConfig.inferenceDefaults());
        int reluNodeId = nodeId(compiled, Operation.OpType.RELU);
        int expNodeId = nodeId(compiled, Operation.OpType.EXP);
        int logNodeId = nodeId(compiled, Operation.OpType.LOG);
        PreparedMetalExecutable executable = (PreparedMetalExecutable) execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .map(step -> testsupport.MetadataArtifacts.acceleratorExecutable(step.metadata()))
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
                        && entry.reason() == runtime.contract.CpuMaterializationReason.CPU_CONSUMER));
        PreparedExecution required = compiled.prepare(runtimeWithRequiredAcceleratorBuffer(ComputeBackend.GPU_METAL));
        assertTrue(required.forwardSteps().stream()
                .anyMatch(step -> step.metadata().backend() == ComputeBackend.GPU_METAL
                        && testsupport.MetadataArtifacts.acceleratorExecutable(step.metadata()) != null));
    }

    @Test
    void cudaElementwiseFusionKeepsInteriorValuesDeviceOwned() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "cudaInteriorA", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 0f, 0f, 1f, 1f, 1f}, new int[]{3, 2}, null, "cudaInteriorB", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor relu = matmul.relu();
        Tensor exp = relu.exp();
        Tensor out = exp.log();
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(matmul, ComputeBackend.GPU_CUDA);
        backendIntentPlan = backendIntentPlan.withBackend(relu, ComputeBackend.GPU_CUDA);
        backendIntentPlan = backendIntentPlan.withBackend(exp, ComputeBackend.GPU_CUDA);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_CUDA);
        CompiledGraph compiled = CompiledGraph.compile(out, CompileConfig.inference(), backendIntentPlan);
        PreparedExecution execution = compiled.prepare(RuntimeConfig.inferenceDefaults());
        int reluNodeId = nodeId(compiled, Operation.OpType.RELU);
        int expNodeId = nodeId(compiled, Operation.OpType.EXP);
        int logNodeId = nodeId(compiled, Operation.OpType.LOG);
        PreparedCudaExecutable executable = (PreparedCudaExecutable) execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_CUDA)
                .map(step -> testsupport.MetadataArtifacts.acceleratorExecutable(step.metadata()))
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
                        && entry.reason() == runtime.contract.CpuMaterializationReason.CPU_CONSUMER));
        PreparedExecution required = compiled.prepare(runtimeWithRequiredAcceleratorBuffer(ComputeBackend.GPU_CUDA));
        assertTrue(required.forwardSteps().stream()
                .anyMatch(step -> step.metadata().backend() == ComputeBackend.GPU_CUDA
                        && testsupport.MetadataArtifacts.acceleratorExecutable(step.metadata()) != null));
    }

    @Test
    void scoredAcceleratorPlanningRecordsCostSummaryAndBoundedFinalists() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{5f, 6f, 7f, 8f}, new int[]{2, 2}, null, "b", DataType.FLOAT32);
        Tensor out = a.matmul(b).relu();

        CompileConfig optimizerConfig = CompileConfig.inference()
                .withBackendPlanning(config.compile.BackendPlanningConfig.autoAccelerator().withOwnershipPlanner(config.compile.RegionOwnershipPlannerStrategy.SCORED));
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
    void scoredAcceleratorPlanningSurfacesExplicitDenseLayoutMaterializationBytes() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "layoutCostA", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{5f, 6f, 7f, 8f}, new int[]{2, 2}, null, "layoutCostB", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor permute = matmul.permute(1, 0);
        Tensor contiguous = permute.contiguous();
        Tensor out = contiguous.relu();

        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();

        backendIntentPlan = backendIntentPlan.withBackend(matmul, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(permute, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(contiguous, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        CompileConfig optimizerConfig = CompileConfig.inference()
                .withBackendPlanning(config.compile.BackendPlanningConfig.autoAccelerator().withOwnershipPlanner(config.compile.RegionOwnershipPlannerStrategy.SCORED));
        CompiledGraph compiled = CompiledGraph.compile(out, optimizerConfig, backendIntentPlan);

        var layoutDecision = compiled.compileTrace().partitionPlanning().decisions().stream()
                .filter(decision -> decision.costSummary() != null)
                .filter(decision -> decision.nodeIds().stream().anyMatch(id -> {
                    CompiledNode node = compiled.program().compiledNodes().get(id);
                    return node.operation() != null && node.operation().opType() == Operation.OpType.CONTIGUOUS;
                }))
                .findFirst()
                .orElseThrow();

        assertTrue(layoutDecision.costSummary().layoutFallbackBytes() >= 16L);
    }

    @Test
    void bfloat16FusedPrepareSkipsCompiledAsmKernel() {
        Tensor a = new Tensor(new double[]{1.0, 2.0, 3.0, 4.0}, new int[]{4}, null, "a", DataType.BFLOAT16);
        Tensor b = new Tensor(new double[]{0.5, 1.5, -2.0, 3.0}, new int[]{4}, null, "b", DataType.BFLOAT16);
        Tensor out = a.add(b).mul(a).sigmoid();

        PreparedExecution execution = CompiledGraph.compile(out, fuseOnlyInferenceConfig())
                .prepare(RuntimeConfig.inferenceDefaults());

        var fusedStep = execution.forwardSteps().stream()
                .filter(step -> testsupport.MetadataArtifacts.fusedExecutable(step.metadata()) != null)
                .findFirst()
                .orElseThrow();

        assertEquals("BFLOAT16", testsupport.MetadataArtifacts.cpuPlan(fusedStep.metadata()).computeContract().storageType().name());
        assertEquals("F32", testsupport.MetadataArtifacts.cpuPlan(fusedStep.metadata()).computeContract().computeType().name());
        assertEquals("CPU_FUSED", testsupport.MetadataArtifacts.cpuPlan(fusedStep.metadata()).computeContract().backend().name());
        assertNotNull(testsupport.MetadataArtifacts.fusedExecutable(fusedStep.metadata()));
    }

    @Test
    void cpuFusedPrepareBuildsMultiNodeStepWithoutPartitionRoles() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{4}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{0.5f, 1.5f, -2f, 3f}, new int[]{4}, null, "b", DataType.FLOAT32);
        Tensor out = a.add(b).mul(a).sigmoid();

        PreparedExecution execution = CompiledGraph.compile(out, fuseOnlyInferenceConfig())
                .prepare(RuntimeConfig.inferenceDefaults());

        List<PreparedExecutionStep> fusedSteps = execution.forwardSteps().stream()
                .filter(step -> testsupport.MetadataArtifacts.fusedExecutable(step.metadata()) != null)
                .toList();
        assertEquals(1, fusedSteps.size());
        PreparedExecutionStep fusedStep = fusedSteps.getFirst();
        assertEquals(Operation.OpType.FUSED, fusedStep.executionOperation().opType());
        assertTrue(fusedStep.orderedNodeIds().size() > 1);
        assertEquals(fusedStep.compiledNode().id(), fusedStep.orderedNodeIds().getLast());
        assertEquals(List.of(fusedStep.compiledNode().id()), fusedStep.boundaryOutputNodeIds());

        Set<Integer> fusedInteriorNodeIds = new java.util.HashSet<>(fusedStep.orderedNodeIds());
        fusedInteriorNodeIds.remove(fusedStep.compiledNode().id());
        assertTrue(execution.forwardSteps().stream()
                        .filter(step -> step != fusedStep)
                        .noneMatch(step -> fusedInteriorNodeIds.contains(step.compiledNode().id())),
                "CPU fused interior nodes must not be emitted as standalone prepared steps.");
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
                        new FusedExecutionPolicy(true)
                ));

        var fusedStep = execution.forwardSteps().stream()
                .filter(step -> testsupport.MetadataArtifacts.fusedExecutable(step.metadata()) != null)
                .findFirst()
                .orElseThrow();

        assertEquals("FLOAT32", testsupport.MetadataArtifacts.cpuPlan(fusedStep.metadata()).computeContract().storageType().name());
        assertEquals("F32", testsupport.MetadataArtifacts.cpuPlan(fusedStep.metadata()).computeContract().computeType().name());
        assertEquals("CPU_FUSED", testsupport.MetadataArtifacts.cpuPlan(fusedStep.metadata()).computeContract().backend().name());
        assertTrue(testsupport.MetadataArtifacts.fusedExecutable(fusedStep.metadata()).getClass().getName().startsWith("backend.cpu.fused.asm."));
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
                        new FusedExecutionPolicy(true)
                ));

        var fusedStep = execution.forwardSteps().stream()
                .filter(step -> testsupport.MetadataArtifacts.fusedExecutable(step.metadata()) != null)
                .findFirst()
                .orElseThrow();

        assertEquals("FLOAT64", testsupport.MetadataArtifacts.cpuPlan(fusedStep.metadata()).computeContract().storageType().name());
        assertEquals("F64", testsupport.MetadataArtifacts.cpuPlan(fusedStep.metadata()).computeContract().computeType().name());
        assertEquals("CPU_FUSED", testsupport.MetadataArtifacts.cpuPlan(fusedStep.metadata()).computeContract().backend().name());
        assertTrue(testsupport.MetadataArtifacts.fusedExecutable(fusedStep.metadata()).getClass().getName().startsWith("backend.cpu.fused.asm."));
    }

    @Test
    void fusedAsmExecutableCacheSeparatesWidthSpecializations() {
        Tensor a1 = new Tensor(new double[]{1.0, 2.0, 3.0, 4.0}, new int[]{4}, null, "a1", DataType.FLOAT64);
        Tensor b1 = new Tensor(new double[]{0.5, 1.5, -2.0, 3.0}, new int[]{4}, null, "b1", DataType.FLOAT64);
        Tensor out1 = a1.add(b1).mul(a1).sigmoid();

        PreparedExecution width1Execution = CompiledGraph.compile(out1, fuseOnlyInferenceConfig())
                .prepare(runtimeWithFusedAsmWidth(1));

        var width1Fused = width1Execution.forwardSteps().stream()
                .filter(step -> testsupport.MetadataArtifacts.fusedExecutable(step.metadata()) != null)
                .findFirst()
                .orElseThrow();

        Tensor a2 = new Tensor(new double[]{1.0, 2.0, 3.0, 4.0}, new int[]{4}, null, "a2", DataType.FLOAT64);
        Tensor b2 = new Tensor(new double[]{0.5, 1.5, -2.0, 3.0}, new int[]{4}, null, "b2", DataType.FLOAT64);
        Tensor out2 = a2.add(b2).mul(a2).sigmoid();

        PreparedExecution width2Execution = CompiledGraph.compile(out2, fuseOnlyInferenceConfig())
                .prepare(runtimeWithFusedAsmWidth(2));

        var width2Fused = width2Execution.forwardSteps().stream()
                .filter(step -> testsupport.MetadataArtifacts.fusedExecutable(step.metadata()) != null)
                .findFirst()
                .orElseThrow();

        assertEquals(1, testsupport.MetadataArtifacts.cpuPlan(width1Fused.metadata()).dispatchHints().vectorWidth());
        assertEquals(2, testsupport.MetadataArtifacts.cpuPlan(width2Fused.metadata()).dispatchHints().vectorWidth());
        assertNotEquals(
                testsupport.MetadataArtifacts.fusedExecutable(width1Fused.metadata()).getClass().getName(),
                testsupport.MetadataArtifacts.fusedExecutable(width2Fused.metadata()).getClass().getName()
        );
        assertTrue(testsupport.MetadataArtifacts.fusedExecutable(width1Fused.metadata()).getClass().getName().endsWith("W1"));
        assertTrue(testsupport.MetadataArtifacts.fusedExecutable(width2Fused.metadata()).getClass().getName().endsWith("W2"));
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
                        new FusedExecutionPolicy(true)
                ));

        var fusedStep = execution.forwardSteps().stream()
                .filter(step -> testsupport.MetadataArtifacts.fusedExecutable(step.metadata()) != null)
                .findFirst()
                .orElseThrow();

        assertTrue(testsupport.MetadataArtifacts.fusedExecutable(fusedStep.metadata()).getClass().getName().startsWith("backend.cpu.fused.asm."));
    }

    @Test
    void bfloat16LinearToReluPreparesCpu1BiasReluSpecialization() {
        Tensor input = new Tensor(new double[32 * 64], new int[]{32, 64}, null, "input", DataType.BFLOAT16);
        Tensor weight = new Tensor(new double[64 * 96], new int[]{64, 96}, null, "weight", DataType.BFLOAT16);
        Tensor bias = new Tensor(new double[96], new int[]{96}, null, "bias", DataType.BFLOAT16);
        Tensor out = input.linear(weight, bias).relu();

        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.inference())
                .prepare(bfloat16BlasRuntime());

        var reluStep = execution.forwardSteps().stream()
                .filter(step -> step.compiledNode().operation() != null && step.compiledNode().operation().opType() == Operation.OpType.RELU)
                .findFirst()
                .orElseThrow();
        Cpu1PreparedArtifact artifact = assertInstanceOf(Cpu1PreparedArtifact.class, reluStep.metadata().artifact());
        Cpu1PreparedMatmulUnit preparedMatmulUnit = artifact.preparedMatmulUnit();

        assertEquals(Operation.OpType.RELU, reluStep.compiledNode().operation().opType());
        assertEquals(2, reluStep.orderedNodeIds().size());
        assertEquals(reluStep.compiledNode().id(), reluStep.orderedNodeIds().getLast());
        assertEquals(DataType.BFLOAT16, preparedMatmulUnit.dataType());
        assertEquals(Cpu1StorageKind.JAVA_ARRAY, preparedMatmulUnit.storageKind());
        assertEquals(Cpu1MatmulRoute.JAVA_SCALAR, preparedMatmulUnit.route());
        assertEquals(Cpu1MatmulKernelId.MATMUL_BF16_DENSE_SCALAR, preparedMatmulUnit.kernelId());
        assertEquals(Cpu1MatmulPostOp.ADD_BIAS_RELU, preparedMatmulUnit.postOp());
    }

    @Test
    void bfloat16MatmulToAddPreparesCpu1BiasSpecialization() {
        Tensor a = new Tensor(new double[64 * 64], new int[]{64, 64}, null, "a", DataType.BFLOAT16);
        Tensor b = new Tensor(new double[64 * 96], new int[]{64, 96}, null, "b", DataType.BFLOAT16);
        Tensor c = new Tensor(new double[64 * 96], new int[]{64, 96}, null, "c", DataType.BFLOAT16);
        Tensor out = a.matmul(b).add(c);

        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.inference())
                .prepare(bfloat16BlasRuntime());

        var addStep = execution.forwardSteps().stream()
                .filter(step -> step.compiledNode().operation() != null && step.compiledNode().operation().opType() == Operation.OpType.ADD)
                .findFirst()
                .orElseThrow();
        Cpu1PreparedArtifact artifact = assertInstanceOf(Cpu1PreparedArtifact.class, addStep.metadata().artifact());
        Cpu1PreparedMatmulUnit preparedMatmulUnit = artifact.preparedMatmulUnit();

        assertEquals(Operation.OpType.ADD, addStep.compiledNode().operation().opType());
        assertEquals(2, addStep.orderedNodeIds().size());
        assertEquals(addStep.compiledNode().id(), addStep.orderedNodeIds().getLast());
        assertEquals(DataType.BFLOAT16, preparedMatmulUnit.dataType());
        assertEquals(Cpu1StorageKind.JAVA_ARRAY, preparedMatmulUnit.storageKind());
        assertEquals(Cpu1MatmulRoute.JAVA_SCALAR, preparedMatmulUnit.route());
        assertEquals(Cpu1MatmulKernelId.MATMUL_BF16_DENSE_SCALAR, preparedMatmulUnit.kernelId());
        assertEquals(Cpu1MatmulPostOp.ADD_BIAS, preparedMatmulUnit.postOp());
    }

    @Test
    void bfloat16MatmulToBroadcastAddToReluPublishesFloatContinuationInInference() {
        Tensor a = new Tensor(new double[16 * 8], new int[]{16, 8}, null, "a", DataType.BFLOAT16);
        Tensor b = new Tensor(new double[8 * 12], new int[]{8, 12}, null, "b", DataType.BFLOAT16);
        Tensor bias = new Tensor(new double[12], new int[]{12}, null, "bias", DataType.BFLOAT16);
        Tensor out = a.matmul(b).add(bias).relu();

        CompiledGraph compiled = CompiledGraph.compile(out, CompileConfig.cpuOnlyBaseline());
        PreparedExecution execution = compiled.prepare(bfloat16BlasRuntime());

        var matmulStep = execution.forwardSteps().stream()
                .filter(step -> step.compiledNode().operation() != null && step.compiledNode().operation().opType() == Operation.OpType.MATMUL)
                .findFirst()
                .orElseThrow();
        var addStep = execution.forwardSteps().stream()
                .filter(step -> step.compiledNode().operation() != null && step.compiledNode().operation().opType() == Operation.OpType.ADD)
                .findFirst()
                .orElseThrow();

        assertTrue(testsupport.MetadataArtifacts.cpuPlan(matmulStep.metadata()).publishFloatContinuation());
        assertTrue(testsupport.MetadataArtifacts.cpuPlan(addStep.metadata()).publishFloatContinuation());
    }

    @Test
    void float64MatmulPrepareBuildsBlasExecutableWhenEligible() {
        Tensor a = new Tensor(new double[64 * 64], new int[]{64, 64}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[64 * 96], new int[]{64, 96}, null, "b", DataType.FLOAT64);
        Tensor out = a.matmul(b);

        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline())
                .prepare(bfloat16BlasRuntime());

        var matmulStep = execution.forwardSteps().stream()
                .filter(step -> step.compiledNode().operation() != null && step.compiledNode().operation().opType() == Operation.OpType.MATMUL)
                .findFirst()
                .orElseThrow();

        assertNotNull(testsupport.MetadataArtifacts.cpuPlan(matmulStep.metadata()).matMulExecutable());
        assertEquals("F64BlasMatMulExecutable", testsupport.MetadataArtifacts.cpuPlan(matmulStep.metadata()).matMulExecutable().getClass().getSimpleName());
    }

    @Test
    void float32MatmulPrepareBuildsJavaExecutableWhenBlasDisabled() {
        Tensor a = new Tensor(new float[64 * 64], new int[]{64, 64}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[64 * 96], new int[]{64, 96}, null, "b", DataType.FLOAT32);
        Tensor out = a.matmul(b);

        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.cpuOnlyBaseline())
                .prepare(RuntimeConfig.inferenceDefaults());

        var matmulStep = execution.forwardSteps().stream()
                .filter(step -> step.compiledNode().operation() != null && step.compiledNode().operation().opType() == Operation.OpType.MATMUL)
                .findFirst()
                .orElseThrow();

        assertNotNull(testsupport.MetadataArtifacts.cpuPlan(matmulStep.metadata()).matMulExecutable());
        assertEquals("F32JavaMatMulExecutable", testsupport.MetadataArtifacts.cpuPlan(matmulStep.metadata()).matMulExecutable().getClass().getSimpleName());
    }

    @Test
    void bfloat16MatmulToFusedNumericChainPublishesFloatContinuationInInference() {
        assumeTrue(OpenBlasRuntime.isBFloat16ToFloatGemmAvailable(), "OpenBLAS SBGEMM is unavailable");

        Tensor a = new Tensor(new double[64 * 64], new int[]{64, 64}, null, "a", DataType.BFLOAT16);
        Tensor b = new Tensor(new double[64 * 96], new int[]{64, 96}, null, "b", DataType.BFLOAT16);
        Tensor out = a.matmul(b).relu().abs().clampMax(1.0);

        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.inference())
                .prepare(bfloat16BlasRuntime());

        var matmulStep = execution.forwardSteps().stream()
                .filter(step -> step.compiledNode().operation() != null && step.compiledNode().operation().opType() == Operation.OpType.MATMUL)
                .findFirst()
                .orElseThrow();

        assertEquals("BFLOAT16", testsupport.MetadataArtifacts.cpuPlan(matmulStep.metadata()).computeContract().storageType().name());
        assertEquals("F32", testsupport.MetadataArtifacts.cpuPlan(matmulStep.metadata()).computeContract().computeType().name());
        assertEquals("CPU_MATMUL_BLAS", testsupport.MetadataArtifacts.cpuPlan(matmulStep.metadata()).computeContract().backend().name());
        assertTrue(testsupport.MetadataArtifacts.cpuPlan(matmulStep.metadata()).publishFloatContinuation());
    }

    @Test
    void bfloat16MatmulToNegPublishesFloatContinuationInInference() {
        Tensor a = new Tensor(new double[64 * 64], new int[]{64, 64}, null, "a", DataType.BFLOAT16);
        Tensor b = new Tensor(new double[64 * 96], new int[]{64, 96}, null, "b", DataType.BFLOAT16);
        Tensor out = a.matmul(b).neg();

        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.inference())
                .prepare(bfloat16BlasRuntime());

        var matmulStep = execution.forwardSteps().stream()
                .filter(step -> step.compiledNode().operation() != null && step.compiledNode().operation().opType() == Operation.OpType.MATMUL)
                .findFirst()
                .orElseThrow();

        assertTrue(testsupport.MetadataArtifacts.cpuPlan(matmulStep.metadata()).publishFloatContinuation());
    }

    @Test
    void bfloat16MatmulToMulScalarPublishesFloatContinuationInInference() {
        Tensor a = new Tensor(new double[64 * 64], new int[]{64, 64}, null, "a", DataType.BFLOAT16);
        Tensor b = new Tensor(new double[64 * 96], new int[]{64, 96}, null, "b", DataType.BFLOAT16);
        Tensor out = a.matmul(b).mul(0.5);

        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.inference())
                .prepare(bfloat16BlasRuntime());

        var matmulStep = execution.forwardSteps().stream()
                .filter(step -> step.compiledNode().operation() != null && step.compiledNode().operation().opType() == Operation.OpType.MATMUL)
                .findFirst()
                .orElseThrow();

        assertTrue(testsupport.MetadataArtifacts.cpuPlan(matmulStep.metadata()).publishFloatContinuation());
    }

    @Test
    void bfloat16MatmulToPowPublishesFloatContinuationInInference() {
        Tensor a = new Tensor(new double[64 * 64], new int[]{64, 64}, null, "a", DataType.BFLOAT16);
        Tensor b = new Tensor(new double[64 * 96], new int[]{64, 96}, null, "b", DataType.BFLOAT16);
        Tensor out = a.matmul(b).pow(1.5);

        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.inference())
                .prepare(bfloat16BlasRuntime());

        var matmulStep = execution.forwardSteps().stream()
                .filter(step -> step.compiledNode().operation() != null && step.compiledNode().operation().opType() == Operation.OpType.MATMUL)
                .findFirst()
                .orElseThrow();

        assertTrue(testsupport.MetadataArtifacts.cpuPlan(matmulStep.metadata()).publishFloatContinuation());
    }

    @Test
    void bfloat16MatmulReshapeToNegPublishesFloatContinuationInInference() {
        Tensor a = new Tensor(new double[64 * 64], new int[]{64, 64}, null, "a", DataType.BFLOAT16);
        Tensor b = new Tensor(new double[64 * 96], new int[]{64, 96}, null, "b", DataType.BFLOAT16);
        Tensor out = a.matmul(b).reshape(32, 192).neg();

        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.inference())
                .prepare(bfloat16BlasRuntime());

        var matmulStep = execution.forwardSteps().stream()
                .filter(step -> step.compiledNode().operation() != null && step.compiledNode().operation().opType() == Operation.OpType.MATMUL)
                .findFirst()
                .orElseThrow();

        assertTrue(testsupport.MetadataArtifacts.cpuPlan(matmulStep.metadata()).publishFloatContinuation());
    }

    @Test
    void bfloat16MatmulToWhereToReluPublishesFloatContinuationInInference() {
        Tensor a = new Tensor(new double[8 * 8], new int[]{8, 8}, null, "a", DataType.BFLOAT16);
        Tensor b = new Tensor(new double[8 * 12], new int[]{8, 12}, null, "b", DataType.BFLOAT16);
        Tensor mask = new Tensor(new byte[8], new int[]{8, 1}, null, "mask", DataType.BOOL);
        Tensor fill = Tensor.scalar(-1.0, DataType.BFLOAT16);
        Tensor out = Tensor.where(mask, a.matmul(b), fill).relu();

        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.cpuOnlyBaseline())
                .prepare(bfloat16BlasRuntime());

        var matmulStep = execution.forwardSteps().stream()
                .filter(step -> step.compiledNode().operation() != null && step.compiledNode().operation().opType() == Operation.OpType.MATMUL)
                .findFirst()
                .orElseThrow();
        var whereStep = execution.forwardSteps().stream()
                .filter(step -> step.compiledNode().operation() != null && step.compiledNode().operation().opType() == Operation.OpType.WHERE)
                .findFirst()
                .orElseThrow();

        assertTrue(testsupport.MetadataArtifacts.cpuPlan(matmulStep.metadata()).publishFloatContinuation());
        assertTrue(testsupport.MetadataArtifacts.cpuPlan(whereStep.metadata()).publishFloatContinuation());
    }

    @Test
    void gpuMetalPartitionPrepareBuildsSingleRegionStepForMatmulAddReluChain() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{3, 2}, null, "b", DataType.FLOAT32);
        Tensor bias = new Tensor(new float[]{1f, -1f}, new int[]{2}, null, "bias", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor add = matmul.add(bias);
        Tensor out = add.relu();

        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();

        backendIntentPlan = backendIntentPlan.withBackend(matmul, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(add, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        CompiledGraph compiled = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline(), backendIntentPlan);
        PreparedExecution execution = compiled.prepare(RuntimeConfig.inferenceDefaults());

        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .toList();
        assertEquals(1, gpuSteps.size());
        assertEquals(1, execution.prepareTrace().backendSelection().selectedCount());
        var anchor = gpuSteps.getFirst();
        assertEquals(ComputeBackend.GPU_METAL, anchor.metadata().backend());
        assertTrue(anchor.orderedNodeIds().size() > 1);
        assertEquals(anchor.compiledNode().id(), anchor.orderedNodeIds().getLast());
        assertEquals(List.of(anchor.compiledNode().id()), anchor.boundaryOutputNodeIds());
        assertNotNull(testsupport.MetadataArtifacts.acceleratorExecutable(anchor.metadata()));
        assertTrue(testsupport.MetadataArtifacts.acceleratorExecutable(anchor.metadata()) instanceof PreparedMetalExecutable);
        PreparedMetalExecutable executable = (PreparedMetalExecutable) testsupport.MetadataArtifacts.acceleratorExecutable(anchor.metadata());
        assertNotNull(executable.bridgeContext());
        assertNotNull(executable.bridgeExecutable());
    }

    @Test
    void gpuMetalMockPartitionExecutionMatchesCpuForMatmulAddReluChain() {
        Tensor cpuA = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "cpuA", DataType.FLOAT32);
        Tensor cpuB = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{3, 2}, null, "cpuB", DataType.FLOAT32);
        Tensor cpuBias = new Tensor(new float[]{1f, -1f}, new int[]{2}, null, "cpuBias", DataType.FLOAT32);
        Tensor cpuOut = cpuA.matmul(cpuB).add(cpuBias).relu();

        CompiledGraph.compile(cpuOut, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{3, 2}, null, "b", DataType.FLOAT32);
        Tensor bias = new Tensor(new float[]{1f, -1f}, new int[]{2}, null, "bias", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor add = matmul.add(bias);
        Tensor out = add.relu();

        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();

        backendIntentPlan = backendIntentPlan.withBackend(matmul, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(add, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline(), backendIntentPlan)
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 1e-6);
    }

    @Test
    void gpuMetalSingleMatmulCanExecuteThroughExplicitAppleShim() {
        String explicitLib = System.getProperty("synaptik.metal.mps.lib");
        assumeTrue(explicitLib != null && !explicitLib.isBlank());

        Tensor cpuA = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "cpuA", DataType.FLOAT32);
        Tensor cpuB = new Tensor(new float[]{7f, 8f, 9f, 10f, 11f, 12f}, new int[]{3, 2}, null, "cpuB", DataType.FLOAT32);
        Tensor cpuOut = cpuA.matmul(cpuB);
        CompiledGraph.compile(cpuOut, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{7f, 8f, 9f, 10f, 11f, 12f}, new int[]{3, 2}, null, "b", DataType.FLOAT32);
        Tensor out = a.matmul(b);
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        CompiledGraph compiled = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline(), backendIntentPlan);
        PreparedExecution execution = compiled.prepare(RuntimeConfig.inferenceDefaults());

        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .toList();
        assertEquals(1, gpuSteps.size());
        PreparedMetalExecutable executable = (PreparedMetalExecutable) testsupport.MetadataArtifacts.acceleratorExecutable(gpuSteps.getFirst().metadata());
        assumeTrue(executable.bridgeContext().available());
        assumeTrue(executable.bridgeExecutable().available());

        execution.execute(ExecutionMode.FORWARD);

        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 1e-5);
    }

    @Test
    void gpuMetalBfloat16SdpaCanExecuteThroughExplicitAppleShim() {
        String explicitLib = System.getProperty("synaptik.metal.mps.lib");
        assumeTrue(explicitLib != null && !explicitLib.isBlank());

        Tensor cpuQ = new Tensor(new double[]{
                1d, 0d,
                0d, 1d
        }, new int[]{1, 2, 2}, null, "cpuBf16SdpaQ", DataType.BFLOAT16);
        Tensor cpuK = new Tensor(new double[]{
                1d, 0d,
                0d, 1d
        }, new int[]{1, 2, 2}, null, "cpuBf16SdpaK", DataType.BFLOAT16);
        Tensor cpuV = new Tensor(new double[]{
                10d, 1d,
                1d, 10d
        }, new int[]{1, 2, 2}, null, "cpuBf16SdpaV", DataType.BFLOAT16);
        Tensor cpuOut = cpuQ.scaledDotProductAttention(cpuK, cpuV, tensor.options.AttentionOptions.defaults().withScale(0.5));
        CompiledGraph.compile(cpuOut, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        Tensor q = new Tensor(new double[]{
                1d, 0d,
                0d, 1d
        }, new int[]{1, 2, 2}, null, "bf16SdpaQ", DataType.BFLOAT16);
        Tensor k = new Tensor(new double[]{
                1d, 0d,
                0d, 1d
        }, new int[]{1, 2, 2}, null, "bf16SdpaK", DataType.BFLOAT16);
        Tensor v = new Tensor(new double[]{
                10d, 1d,
                1d, 10d
        }, new int[]{1, 2, 2}, null, "bf16SdpaV", DataType.BFLOAT16);
        Tensor out = q.scaledDotProductAttention(k, v, tensor.options.AttentionOptions.defaults().withScale(0.5));
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        CompiledGraph compiled = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline(), backendIntentPlan);
        PreparedExecution execution = compiled.prepare(RuntimeConfig.inferenceDefaults());

        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .toList();
        assertEquals(1, gpuSteps.size());
        PreparedMetalExecutable executable = (PreparedMetalExecutable) testsupport.MetadataArtifacts.acceleratorExecutable(gpuSteps.getFirst().metadata());
        assumeTrue(executable.bridgeContext().available());
        assumeTrue(executable.bridgeExecutable().available());

        execution.execute(ExecutionMode.FORWARD);

        assertFalse(executable.lastExecutionStats().usedCpuFallback(), executable.lastExecutionStats()::fallbackReason);
        assertEquals(3, executable.plan().lowering().dagSpec().externalInputs().size(),
                () -> "dag=" + executable.plan().lowering().dagSpec());
        assertTrue(executable.plan().lowering().dagSpec().nodes().size() > 1,
                () -> "dag=" + executable.plan().lowering().dagSpec());
        assertArrayEquals(
                cpuOut.toDoubleArrayCopy(),
                out.toDoubleArrayCopy(),
                3e-2,
                () -> "stats=" + executable.lastExecutionStats()
                        + " dag=" + executable.plan().lowering().dagSpec()
        );
    }

    @Test
    void gpuMetalDenseCrossEntropyAndNllCanExecuteThroughExplicitAppleShim() {
        String explicitLib = System.getProperty("synaptik.metal.mps.lib");
        assumeTrue(explicitLib != null && !explicitLib.isBlank());

        Tensor cpuLogits = new Tensor(new float[]{
                1f, 2f, 3f,
                1f, 0f, -1f
        }, new int[]{2, 3}, null, "cpuDenseLossLogits", DataType.FLOAT32);
        Tensor cpuCeTargets = new Tensor(new float[]{
                0f, 0f, 1f,
                1f, 0f, 0f
        }, new int[]{2, 3}, null, "cpuDenseLossCeTargets", DataType.FLOAT32);
        Tensor cpuCe = cpuLogits.crossEntropyLoss(cpuCeTargets, 1);
        CompiledGraph.compile(cpuCe, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        Tensor logits = new Tensor(new float[]{
                1f, 2f, 3f,
                1f, 0f, -1f
        }, new int[]{2, 3}, null, "metalDenseLossLogits", DataType.FLOAT32);
        Tensor ceTargets = new Tensor(new float[]{
                0f, 0f, 1f,
                1f, 0f, 0f
        }, new int[]{2, 3}, null, "metalDenseLossCeTargets", DataType.FLOAT32);
        Tensor ce = logits.crossEntropyLoss(ceTargets, 1);
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(ce, ComputeBackend.GPU_METAL);
        PreparedExecution ceExecution = CompiledGraph.compile(ce, CompileConfig.noGraphOptimizationBaseline(), backendIntentPlan)
                .prepare(RuntimeConfig.inferenceDefaults());
        PreparedMetalExecutable ceExecutable = (PreparedMetalExecutable) ceExecution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .map(step -> testsupport.MetadataArtifacts.acceleratorExecutable(step.metadata()))
                .findFirst()
                .orElseThrow();
        assumeTrue(ceExecutable.bridgeContext().available());
        assumeTrue(ceExecutable.bridgeExecutable().available());
        ceExecution.execute(ExecutionMode.FORWARD);

        Tensor cpuLogProbs = new Tensor(new float[]{
                -0.16984604f, -2.169846f, -3.169846f,
                -1.407606f, -0.40760595f, -2.407606f
        }, new int[]{2, 3}, null, "cpuDenseLossLogProbs", DataType.FLOAT32);
        Tensor cpuNllTargets = new Tensor(new float[]{
                1f, 0f, 0f,
                0f, 1f, 0f
        }, new int[]{2, 3}, null, "cpuDenseLossNllTargets", DataType.FLOAT32);
        Tensor cpuNll = cpuLogProbs.nllLoss(cpuNllTargets, 1);
        CompiledGraph.compile(cpuNll, CompileConfig.noGraphOptimizationBaseline(), backendIntentPlan)
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        Tensor logProbs = new Tensor(new float[]{
                -0.16984604f, -2.169846f, -3.169846f,
                -1.407606f, -0.40760595f, -2.407606f
        }, new int[]{2, 3}, null, "metalDenseLossLogProbs", DataType.FLOAT32);
        Tensor nllTargets = new Tensor(new float[]{
                1f, 0f, 0f,
                0f, 1f, 0f
        }, new int[]{2, 3}, null, "metalDenseLossNllTargets", DataType.FLOAT32);
        Tensor nll = logProbs.nllLoss(nllTargets, 1);
        backendIntentPlan = backendIntentPlan.withBackend(nll, ComputeBackend.GPU_METAL);
        PreparedExecution nllExecution = CompiledGraph.compile(nll, CompileConfig.noGraphOptimizationBaseline(), backendIntentPlan)
                .prepare(RuntimeConfig.inferenceDefaults());
        PreparedMetalExecutable nllExecutable = (PreparedMetalExecutable) nllExecution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .map(step -> testsupport.MetadataArtifacts.acceleratorExecutable(step.metadata()))
                .findFirst()
                .orElseThrow();
        assumeTrue(nllExecutable.bridgeContext().available());
        assumeTrue(nllExecutable.bridgeExecutable().available());
        nllExecution.execute(ExecutionMode.FORWARD);

        assertFalse(ceExecutable.lastExecutionStats().usedCpuFallback());
        assertFalse(nllExecutable.lastExecutionStats().usedCpuFallback());
        assertArrayEquals(cpuCe.toDoubleArrayCopy(), ce.toDoubleArrayCopy(), 1e-5);
        assertArrayEquals(cpuNll.toDoubleArrayCopy(), nll.toDoubleArrayCopy(), 1e-5);
    }

    @Test
    void gpuMetalDenseCrossEntropyForwardStaysGpuOwnedInTrainingUntilGradientPublication() {
        Tensor cpuLogits = new Tensor(new float[]{
                1f, 2f, 3f,
                1f, 0f, -1f
        }, new int[]{2, 3}, null, "cpuTrainingDenseLossLogits", DataType.FLOAT32);
        cpuLogits.setRequiresGrad(true);
        Tensor cpuTargets = new Tensor(new float[]{
                0f, 0f, 1f,
                1f, 0f, 0f
        }, new int[]{2, 3}, null, "cpuTrainingDenseLossTargets", DataType.FLOAT32);
        Tensor cpuLoss = cpuLogits.crossEntropyLoss(cpuTargets, 1);
        CompiledGraph.compile(cpuLoss, CompileConfig.training())
                .prepare(RuntimeConfig.trainingDefaults()).execute(ExecutionMode.FORWARD_BACKWARD);

        Tensor logits = new Tensor(new float[]{
                1f, 2f, 3f,
                1f, 0f, -1f
        }, new int[]{2, 3}, null, "metalTrainingDenseLossLogits", DataType.FLOAT32);
        logits.setRequiresGrad(true);
        Tensor targets = new Tensor(new float[]{
                0f, 0f, 1f,
                1f, 0f, 0f
        }, new int[]{2, 3}, null, "metalTrainingDenseLossTargets", DataType.FLOAT32);
        Tensor loss = logits.crossEntropyLoss(targets, 1);
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(loss, ComputeBackend.GPU_METAL);
        CompiledGraph compiled = CompiledGraph.compile(loss, CompileConfig.training(), backendIntentPlan);
        PreparedExecution execution = compiled.prepare(RuntimeConfig.trainingDefaults());
        int lossNodeId = nodeId(compiled, Operation.OpType.CROSS_ENTROPY_LOSS);

        assertTrue(hasSelectedAcceleratorDecisionFor(execution, ComputeBackend.GPU_METAL, lossNodeId));
        assertTrue(execution.forwardSteps().stream()
                .anyMatch(step -> step.compiledNode().id() == lossNodeId && step.metadata().backend() == ComputeBackend.GPU_METAL));

        var trace = execution.executeTraced(ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(cpuLogits.getGradient().toDoubleArrayCopy(), logits.getGradient().toDoubleArrayCopy(), 1e-5);
        assertTrue(trace.steps().stream()
                .anyMatch(step -> step.backend().equals("GPU_METAL") && step.opType().equals("CROSS_ENTROPY_LOSS")));
        assertFalse(trace.cpuMaterializations().stream()
                .anyMatch(materialization -> materialization.nodeId() == lossNodeId
                        && materialization.reason() == runtime.contract.CpuMaterializationReason.CPU_CONSUMER));
        assertFalse(trace.cpuMaterializations().stream()
                .anyMatch(materialization -> materialization.reason() == runtime.contract.CpuMaterializationReason.CPU_FALLBACK));
    }

    @Test
    void gpuMetalIndexTargetLossGradientStaysDeviceOwnedInTraining() {
        Tensor logits = new Tensor(new float[]{
                1f, 2f, 3f,
                1f, 0f, -1f
        }, new int[]{2, 3}, null, "metalIndexLossTrainingLogits", DataType.FLOAT32);
        logits.setRequiresGrad(true);
        Tensor targetIndices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "metalIndexLossTrainingTargets", DataType.INT32);
        Tensor loss = logits.crossEntropyLossFromIndices(targetIndices, 1);
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(loss, ComputeBackend.GPU_METAL);
        CompiledGraph compiled = CompiledGraph.compile(loss, CompileConfig.training(), backendIntentPlan);
        PreparedExecution execution = compiled.prepare(RuntimeConfig.trainingDefaults());
        int forwardLossNodeId = nodeId(compiled, Operation.OpType.CROSS_ENTROPY_LOSS_INDICES);
        String forwardReason = MetalPartitionSupport.plannerUnsupportedReason(compiledNode(compiled, forwardLossNodeId), planningContext(compiled));
        Integer gradNodeId = compiled.program().compiledNodes().stream()
                .filter(node -> node.operation() != null
                        && node.operation().opType() == Operation.OpType.CROSS_ENTROPY_LOSS_INDICES_GRAD)
                .map(graph.model.CompiledNode::id)
                .findFirst()
                .orElse(null);

        assertTrue(hasSelectedAcceleratorDecisionFor(execution, ComputeBackend.GPU_METAL, forwardLossNodeId));
        assertTrue(execution.forwardSteps().stream()
                .anyMatch(step -> step.compiledNode().id() == forwardLossNodeId && step.metadata().backend() == ComputeBackend.GPU_METAL));
        assertTrue(forwardReason.isBlank(), forwardReason);
        if (gradNodeId != null) {
            String gradReason = MetalPartitionSupport.plannerUnsupportedReason(compiledNode(compiled, gradNodeId), planningContext(compiled));
            assertTrue(hasSelectedAcceleratorDecisionFor(execution, ComputeBackend.GPU_METAL, gradNodeId));
            assertTrue(execution.backwardSteps().stream()
                    .anyMatch(step -> step.compiledNode().id() == gradNodeId && step.metadata().backend() == ComputeBackend.GPU_METAL));
            assertTrue(gradReason.isBlank(), gradReason);
        }
    }

    @Test
    void gpuCudaDirectUnmaskedSdpaFallsBackWithCapabilityMissingReason() {
        Tensor cpuQ = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "cpuCudaSdpaQ", DataType.FLOAT32);
        Tensor cpuK = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "cpuCudaSdpaK", DataType.FLOAT32);
        Tensor cpuV = new Tensor(new float[]{10f, 1f, 1f, 10f}, new int[]{1, 2, 2}, null, "cpuCudaSdpaV", DataType.FLOAT32);
        Tensor cpuOut = cpuQ.scaledDotProductAttention(cpuK, cpuV, AttentionOptions.defaults().withScale(0.5));
        CompiledGraph.compile(cpuOut, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        Tensor q = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "cudaSdpaQ", DataType.FLOAT32);
        Tensor k = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "cudaSdpaK", DataType.FLOAT32);
        Tensor v = new Tensor(new float[]{10f, 1f, 1f, 10f}, new int[]{1, 2, 2}, null, "cudaSdpaV", DataType.FLOAT32);
        Tensor out = q.scaledDotProductAttention(k, v, AttentionOptions.defaults().withScale(0.5));
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_CUDA);
        CompiledGraph compiled = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline(), backendIntentPlan);
        PreparedExecution execution = compiled.prepare(RuntimeConfig.inferenceDefaults());
        assertFalse(compiled.program().compiledNodes().stream()
                .anyMatch(node -> node.operation() != null
                        && node.operation().opType() == Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION));

        execution.execute(ExecutionMode.FORWARD);

        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 1e-5);
    }

    @Test
    void gpuMetalDirectUnmaskedSdpaUsesPreparedMetalExecutableWhenAvailable() {
        Tensor cpuQ = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "cpuQ", DataType.FLOAT32);
        Tensor cpuK = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "cpuK", DataType.FLOAT32);
        Tensor cpuV = new Tensor(new float[]{10f, 1f, 1f, 10f}, new int[]{1, 2, 2}, null, "cpuV", DataType.FLOAT32);
        Tensor cpuOut = cpuQ.scaledDotProductAttention(cpuK, cpuV, tensor.options.AttentionOptions.defaults().withScale(0.5));
        CompiledGraph.compile(cpuOut, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        Tensor q = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "q", DataType.FLOAT32);
        Tensor k = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "k", DataType.FLOAT32);
        Tensor v = new Tensor(new float[]{10f, 1f, 1f, 10f}, new int[]{1, 2, 2}, null, "v", DataType.FLOAT32);
        Tensor out = q.scaledDotProductAttention(k, v, tensor.options.AttentionOptions.defaults().withScale(0.5));
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline(), backendIntentPlan)
                .prepare(RuntimeConfig.inferenceDefaults());
        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .toList();
        if (!gpuSteps.isEmpty()) {
            assertEquals(1, gpuSteps.size());
            PreparedMetalExecutable executable = (PreparedMetalExecutable) testsupport.MetadataArtifacts.acceleratorExecutable(gpuSteps.getFirst().metadata());
            assertFalse(executable.plan().lowering().dagSpec().nodes().stream()
                    .anyMatch(node -> node.type() == backend.accelerator.dag.AcceleratorDagNodeType.SDPA));
            assertTrue(executable.plan().lowering().dagSpec().nodes().stream()
                    .anyMatch(node -> node.type() == backend.accelerator.dag.AcceleratorDagNodeType.MATMUL));
        }

        execution.execute(ExecutionMode.FORWARD);

        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 1e-5);
    }

    @Test
    void gpuMetalDirectMaskedSdpaCanPrepareMaskedNativeDagWhenAvailable() {
        Tensor cpuQ = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "cpuQMask", DataType.FLOAT32);
        Tensor cpuK = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "cpuKMask", DataType.FLOAT32);
        Tensor cpuV = new Tensor(new float[]{10f, 1f, 1f, 10f}, new int[]{1, 2, 2}, null, "cpuVMask", DataType.FLOAT32);
        Tensor cpuMask = new Tensor(new byte[]{1, 0, 1, 1}, new int[]{1, 2, 2}, null, "cpuMask", DataType.BOOL);
        Tensor cpuOut = cpuQ.scaledDotProductAttention(cpuK, cpuV, cpuMask, tensor.options.AttentionOptions.defaults().withScale(0.5));
        CompiledGraph.compile(cpuOut, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        Tensor q = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "qMask", DataType.FLOAT32);
        Tensor k = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "kMask", DataType.FLOAT32);
        Tensor v = new Tensor(new float[]{10f, 1f, 1f, 10f}, new int[]{1, 2, 2}, null, "vMask", DataType.FLOAT32);
        Tensor mask = new Tensor(new byte[]{1, 0, 1, 1}, new int[]{1, 2, 2}, null, "mask", DataType.BOOL);
        Tensor out = q.scaledDotProductAttention(k, v, mask, tensor.options.AttentionOptions.defaults().withScale(0.5));
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline(), backendIntentPlan)
                .prepare(RuntimeConfig.inferenceDefaults());
        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .toList();
        if (!gpuSteps.isEmpty()) {
            assertEquals(1, gpuSteps.size());
            PreparedMetalExecutable executable = (PreparedMetalExecutable) testsupport.MetadataArtifacts.acceleratorExecutable(gpuSteps.getFirst().metadata());
            assertFalse(executable.plan().lowering().dagSpec().nodes().stream()
                    .anyMatch(node -> node.type() == backend.accelerator.dag.AcceleratorDagNodeType.SDPA));
            assertTrue(executable.plan().lowering().dagSpec().nodes().stream()
                    .anyMatch(node -> node.type() == backend.accelerator.dag.AcceleratorDagNodeType.MATMUL));
        }

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
        CompiledGraph.compile(cpuOut, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{7f, 8f, 9f, 10f, 11f, 12f}, new int[]{2, 3}, null, "b", DataType.FLOAT32);
        Tensor out = a.matmul(b.transpose());
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline(), backendIntentPlan)
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .toList();
        assertEquals(1, gpuSteps.size());
        PreparedMetalExecutable executable = (PreparedMetalExecutable) testsupport.MetadataArtifacts.acceleratorExecutable(gpuSteps.getFirst().metadata());
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
        CompiledGraph.compile(cpuOut, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{4}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{5f, 6f, 7f, 8f}, new int[]{4}, null, "b", DataType.FLOAT32);
        Tensor add = a.add(b);
        Tensor relu = add.relu();
        Tensor out = relu.exp();

        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();

        backendIntentPlan = backendIntentPlan.withBackend(add, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(relu, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline(), backendIntentPlan)
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .toList();
        assertEquals(1, gpuSteps.size());
        PreparedMetalExecutable executable = (PreparedMetalExecutable) testsupport.MetadataArtifacts.acceleratorExecutable(gpuSteps.getFirst().metadata());
        assertEquals(3, executable.plan().lowering().dagSpec().nodes().size());

        execution.execute(ExecutionMode.FORWARD);

        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 3e-3);
    }

    @Test
    void gpuMetalPureElementwiseChainUsesMpsGraphRegionLoweringWhenOptimizerRegionsExist() {
        Tensor cpuA = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{4}, null, "cpuAOpt", DataType.FLOAT32);
        Tensor cpuB = new Tensor(new float[]{5f, 6f, 7f, 8f}, new int[]{4}, null, "cpuBOpt", DataType.FLOAT32);
        Tensor cpuOut = cpuA.add(cpuB).relu().exp();
        CompiledGraph.compile(cpuOut, CompileConfig.inference())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{4}, null, "aOpt", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{5f, 6f, 7f, 8f}, new int[]{4}, null, "bOpt", DataType.FLOAT32);
        Tensor add = a.add(b);
        Tensor relu = add.relu();
        Tensor out = relu.exp();

        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();

        backendIntentPlan = backendIntentPlan.withBackend(add, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(relu, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.inference(), backendIntentPlan)
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .toList();
        assertEquals(1, gpuSteps.size());
        PreparedMetalExecutable executable = (PreparedMetalExecutable) testsupport.MetadataArtifacts.acceleratorExecutable(gpuSteps.getFirst().metadata());
        assertEquals(backend.lowering.LoweringFamily.METAL_GRAPH_REGION, executable.loweringFamily());
        assertTrue(executable.plan().manifest().fusedSubpatterns().stream()
                .anyMatch(subpattern -> subpattern.patternType() == GpuCompoundPatternType.ELEMENTWISE_CHAIN));

        execution.execute(ExecutionMode.FORWARD);

        assertArrayEquals(cpuOut.toDoubleArrayCopy(), out.toDoubleArrayCopy(), 3e-3);
    }

    @Test
    void gpuCudaPureElementwiseChainCanPrepareAndExecuteThroughFallback() {
        Tensor cpuA = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{4}, null, "cpuCudaA", DataType.FLOAT32);
        Tensor cpuB = new Tensor(new float[]{5f, 6f, 7f, 8f}, new int[]{4}, null, "cpuCudaB", DataType.FLOAT32);
        Tensor cpuOut = cpuA.add(cpuB).relu().exp();
        CompiledGraph.compile(cpuOut, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{4}, null, "cudaA", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{5f, 6f, 7f, 8f}, new int[]{4}, null, "cudaB", DataType.FLOAT32);
        Tensor add = a.add(b);
        Tensor relu = add.relu();
        Tensor out = relu.exp();

        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();

        backendIntentPlan = backendIntentPlan.withBackend(add, ComputeBackend.GPU_CUDA);
        backendIntentPlan = backendIntentPlan.withBackend(relu, ComputeBackend.GPU_CUDA);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_CUDA);
        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.inference(), backendIntentPlan)
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_CUDA)
                .toList();
        assertEquals(1, gpuSteps.size());
        PreparedCudaExecutable executable = (PreparedCudaExecutable) testsupport.MetadataArtifacts.acceleratorExecutable(gpuSteps.getFirst().metadata());
        assertEquals(backend.lowering.LoweringFamily.CUDA_GRAPH_REGION, executable.loweringFamily());

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
        CompiledGraph.compile(cpuLoss, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.trainingDefaults()).execute(ExecutionMode.FORWARD_BACKWARD);
        double[] expectedGradA = cpuA.getGradient().toDoubleArrayCopy().clone();
        double[] expectedGradB = cpuB.getGradient().toDoubleArrayCopy().clone();

        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{7f, 8f, 9f, 10f, 11f, 12f}, new int[]{3, 2}, null, "b", DataType.FLOAT32);
        a.setRequiresGrad(true);
        b.setRequiresGrad(true);
        Tensor matmul = a.matmul(b);
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(matmul, ComputeBackend.GPU_METAL);
        Tensor loss = matmul.sum();

        PreparedExecution execution = CompiledGraph.compile(loss, CompileConfig.noGraphOptimizationBaseline(), backendIntentPlan)
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
        CompiledGraph.compile(cpuLoss, CompileConfig.training())
                .prepare(RuntimeConfig.trainingDefaults()).execute(ExecutionMode.FORWARD_BACKWARD);
        double[] expectedGrad = cpuInput.getGradient().toDoubleArrayCopy().clone();

        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "input", DataType.FLOAT32);
        input.setRequiresGrad(true);
        Tensor softmax = input.exp().softmax(1);
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(softmax, ComputeBackend.GPU_METAL);
        Tensor loss = softmax.sum();

        PreparedExecution execution = CompiledGraph.compile(loss, CompileConfig.training(), backendIntentPlan)
                .prepare(RuntimeConfig.trainingDefaults());

        assertFalse(execution.backwardSteps().stream()
                .anyMatch(step -> step.compiledNode().operation() != null
                        && step.compiledNode().operation().opType() == Operation.OpType.SOFTMAX_GRAD));
        assertTrue(execution.backwardSteps().stream()
                .anyMatch(step -> step.metadata().backend() == ComputeBackend.GPU_METAL));

        execution.execute(ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(expectedGrad, input.getGradient().toDoubleArrayCopy(), 1e-5);
    }

    @Test
    void gpuMetalBackwardLogSoftmaxGradCanPrepareAndMatchCpuGradients() {
        Tensor cpuInput = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "cpuInput", DataType.FLOAT32);
        cpuInput.setRequiresGrad(true);
        Tensor cpuLogSoftmax = cpuInput.exp().logSoftmax(1);
        Tensor cpuLoss = cpuLogSoftmax.sum();
        CompiledGraph.compile(cpuLoss, CompileConfig.training())
                .prepare(RuntimeConfig.trainingDefaults()).execute(ExecutionMode.FORWARD_BACKWARD);
        double[] expectedGrad = cpuInput.getGradient().toDoubleArrayCopy().clone();

        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "input", DataType.FLOAT32);
        input.setRequiresGrad(true);
        Tensor logSoftmax = input.exp().logSoftmax(1);
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(logSoftmax, ComputeBackend.GPU_METAL);
        Tensor loss = logSoftmax.sum();

        PreparedExecution execution = CompiledGraph.compile(loss, CompileConfig.training(), backendIntentPlan)
                .prepare(RuntimeConfig.trainingDefaults());

        assertFalse(execution.backwardSteps().stream()
                .anyMatch(step -> step.compiledNode().operation() != null
                        && step.compiledNode().operation().opType() == Operation.OpType.LOG_SOFTMAX_GRAD));
        assertTrue(execution.backwardSteps().stream()
                .anyMatch(step -> step.metadata().backend() == ComputeBackend.GPU_METAL));

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
        CompiledGraph.compile(cpuLoss, CompileConfig.training())
                .prepare(RuntimeConfig.trainingDefaults()).execute(ExecutionMode.FORWARD_BACKWARD);
        double[] expectedGrad = cpuInput.getGradient().toDoubleArrayCopy().clone();

        Tensor input = new Tensor(new float[]{
                1f, 1f, 2f,
                3f, 2f, 2f
        }, new int[]{2, 3}, null, "input", DataType.FLOAT32);
        input.setRequiresGrad(true);
        Tensor reduced = input.min(1, true);
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(reduced, ComputeBackend.GPU_METAL);
        Tensor loss = reduced.sum();

        PreparedExecution execution = CompiledGraph.compile(loss, CompileConfig.training(), backendIntentPlan)
                .prepare(RuntimeConfig.trainingDefaults());

        execution.execute(ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(expectedGrad, input.getGradient().toDoubleArrayCopy(), 1e-5);
    }

    @Test
    void gpuMetalBackwardReduceMaxGradCanPrepareAndMatchCpuGradients() {
        Tensor cpuInput = new Tensor(new float[]{1f, 5f, 5f, 2f}, new int[]{4}, null, "cpuInput", DataType.FLOAT32);
        cpuInput.setRequiresGrad(true);
        Tensor cpuReduced = cpuInput.max();
        Tensor cpuLoss = cpuReduced.sum();
        CompiledGraph.compile(cpuLoss, CompileConfig.training())
                .prepare(RuntimeConfig.trainingDefaults()).execute(ExecutionMode.FORWARD_BACKWARD);
        double[] expectedGrad = cpuInput.getGradient().toDoubleArrayCopy().clone();

        Tensor input = new Tensor(new float[]{1f, 5f, 5f, 2f}, new int[]{4}, null, "input", DataType.FLOAT32);
        input.setRequiresGrad(true);
        Tensor reduced = input.max();
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(reduced, ComputeBackend.GPU_METAL);
        Tensor loss = reduced.sum();

        PreparedExecution execution = CompiledGraph.compile(loss, CompileConfig.training(), backendIntentPlan)
                .prepare(RuntimeConfig.trainingDefaults());

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
        CompiledGraph.compile(cpuLoss, CompileConfig.training())
                .prepare(RuntimeConfig.trainingDefaults()).execute(ExecutionMode.FORWARD_BACKWARD);
        double[] expectedGradA = cpuA.getGradient().toDoubleArrayCopy().clone();
        double[] expectedGradB = cpuB.getGradient().toDoubleArrayCopy().clone();

        Tensor a = new Tensor(new float[]{1f, 5f, 3f}, new int[]{3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{2f, 4f, 3f}, new int[]{3}, null, "b", DataType.FLOAT32);
        a.setRequiresGrad(true);
        b.setRequiresGrad(true);
        Tensor min = a.min(b);
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(min, ComputeBackend.GPU_METAL);
        Tensor loss = min.sum();

        PreparedExecution execution = CompiledGraph.compile(loss, CompileConfig.training(), backendIntentPlan)
                .prepare(RuntimeConfig.trainingDefaults());

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
        CompiledGraph.compile(cpuLoss, CompileConfig.training())
                .prepare(RuntimeConfig.trainingDefaults()).execute(ExecutionMode.FORWARD_BACKWARD);
        double[] expectedGradA = cpuA.getGradient().toDoubleArrayCopy().clone();
        double[] expectedGradB = cpuB.getGradient().toDoubleArrayCopy().clone();

        Tensor a = new Tensor(new float[]{1f, 5f, 3f}, new int[]{3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{2f, 4f, 3f}, new int[]{3}, null, "b", DataType.FLOAT32);
        a.setRequiresGrad(true);
        b.setRequiresGrad(true);
        Tensor max = a.max(b);
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(max, ComputeBackend.GPU_METAL);
        Tensor loss = max.sum();

        PreparedExecution execution = CompiledGraph.compile(loss, CompileConfig.training(), backendIntentPlan)
                .prepare(RuntimeConfig.trainingDefaults());

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
        CompiledGraph.compile(cpuLoss, CompileConfig.training())
                .prepare(RuntimeConfig.trainingDefaults()).execute(ExecutionMode.FORWARD_BACKWARD);
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
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(attention, ComputeBackend.GPU_METAL);
        Tensor loss = attention.sum();

        PreparedExecution execution = CompiledGraph.compile(loss, CompileConfig.training(), backendIntentPlan)
                .prepare(RuntimeConfig.trainingDefaults());

        assertFalse(execution.backwardSteps().stream()
                .anyMatch(step -> step.compiledNode().operation() != null
                        && step.compiledNode().operation().opType() == Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION_BACKWARD));

        execution.execute(ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(expectedGradQ, q.getGradient().toDoubleArrayCopy(), 1e-5);
        assertArrayEquals(expectedGradK, k.getGradient().toDoubleArrayCopy(), 1e-5);
        assertArrayEquals(expectedGradV, v.getGradient().toDoubleArrayCopy(), 1e-5);
    }

    @Test
    void gpuMetalBackwardMaskedCausalSdpaCanPrepareAndMatchCpuGradients() {
        Tensor cpuQ = new Tensor(new float[]{
                1f, 0f,
                0f, 1f
        }, new int[]{1, 2, 2}, null, "cpuMaskedQ", DataType.FLOAT32);
        Tensor cpuK = new Tensor(new float[]{
                1f, 0f,
                0f, 1f
        }, new int[]{1, 2, 2}, null, "cpuMaskedK", DataType.FLOAT32);
        Tensor cpuV = new Tensor(new float[]{
                10f, 1f,
                1f, 10f
        }, new int[]{1, 2, 2}, null, "cpuMaskedV", DataType.FLOAT32);
        Tensor cpuMask = new Tensor(new byte[]{1, 1, 1, 0}, new int[]{1, 2, 2}, null, "cpuMaskedSdpaMask", DataType.BOOL);
        cpuQ.setRequiresGrad(true);
        cpuK.setRequiresGrad(true);
        cpuV.setRequiresGrad(true);
        Tensor cpuLoss = cpuQ.scaledDotProductAttention(
                cpuK,
                cpuV,
                cpuMask,
                tensor.options.AttentionOptions.causalDefaults().withScale(1.0)
        ).sum();
        CompiledGraph.compile(cpuLoss, CompileConfig.training())
                .prepare(RuntimeConfig.trainingDefaults()).execute(ExecutionMode.FORWARD_BACKWARD);
        double[] expectedGradQ = cpuQ.getGradient().toDoubleArrayCopy().clone();
        double[] expectedGradK = cpuK.getGradient().toDoubleArrayCopy().clone();
        double[] expectedGradV = cpuV.getGradient().toDoubleArrayCopy().clone();

        Tensor q = new Tensor(new float[]{
                1f, 0f,
                0f, 1f
        }, new int[]{1, 2, 2}, null, "maskedQ", DataType.FLOAT32);
        Tensor k = new Tensor(new float[]{
                1f, 0f,
                0f, 1f
        }, new int[]{1, 2, 2}, null, "maskedK", DataType.FLOAT32);
        Tensor v = new Tensor(new float[]{
                10f, 1f,
                1f, 10f
        }, new int[]{1, 2, 2}, null, "maskedV", DataType.FLOAT32);
        Tensor mask = new Tensor(new byte[]{1, 1, 1, 0}, new int[]{1, 2, 2}, null, "maskedSdpaMask", DataType.BOOL);
        q.setRequiresGrad(true);
        k.setRequiresGrad(true);
        v.setRequiresGrad(true);
        Tensor attention = q.scaledDotProductAttention(
                k,
                v,
                mask,
                tensor.options.AttentionOptions.causalDefaults().withScale(1.0)
        );
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(attention, ComputeBackend.GPU_METAL);
        Tensor loss = attention.sum();

        PreparedExecution execution = CompiledGraph.compile(loss, CompileConfig.training(), backendIntentPlan)
                .prepare(RuntimeConfig.trainingDefaults());

        assertFalse(execution.backwardSteps().stream()
                .anyMatch(step -> step.compiledNode().operation() != null
                        && step.compiledNode().operation().opType() == Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION_BACKWARD));

        execution.execute(ExecutionMode.FORWARD_BACKWARD);

        assertArrayEquals(expectedGradQ, q.getGradient().toDoubleArrayCopy(), 1e-5);
        assertArrayEquals(expectedGradK, k.getGradient().toDoubleArrayCopy(), 1e-5);
        assertArrayEquals(expectedGradV, v.getGradient().toDoubleArrayCopy(), 1e-5);
    }

    @Test
    void gpuMetalSupportedBackwardRowsUseNativeBufferBindingWithExplicitAppleShim() {
        String explicitLib = System.getProperty("synaptik.metal.mps.lib");
        assumeTrue(explicitLib != null && !explicitLib.isBlank());

        Tensor softmaxInput = trainable("metalBackwardTraceSoftmax", 2, 3);
        Tensor softmax = softmaxInput.exp().softmax(1);
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(softmax, ComputeBackend.GPU_METAL);
        assertMetalBackwardBufferBinding(weightedSum(softmax, "metalBackwardTraceSoftmaxWeight"), Operation.OpType.MUL, 1);

        Tensor logSoftmaxInput = trainable("metalBackwardTraceLogSoftmax", 2, 3);
        Tensor logSoftmax = logSoftmaxInput.exp().logSoftmax(1);
        backendIntentPlan = backendIntentPlan.withBackend(logSoftmax, ComputeBackend.GPU_METAL);
        assertMetalBackwardBufferBinding(weightedSum(logSoftmax, "metalBackwardTraceLogSoftmaxWeight"), Operation.OpType.EXP, 1);

        Tensor minInput = trainable("metalBackwardTraceReduceMin", 2, 3);
        Tensor reduceMin = minInput.min(1, true);
        backendIntentPlan = backendIntentPlan.withBackend(reduceMin, ComputeBackend.GPU_METAL);
        Tensor maxInput = trainable("metalBackwardTraceReduceMax", 2, 4);
        Tensor reduceMax = maxInput.max(0, true);
        backendIntentPlan = backendIntentPlan.withBackend(reduceMax, ComputeBackend.GPU_METAL);
        Tensor a = trainable("metalBackwardTraceMinA", 3);
        Tensor b = trainable("metalBackwardTraceMinB", 3);
        Tensor min = a.min(b);
        backendIntentPlan = backendIntentPlan.withBackend(min, ComputeBackend.GPU_METAL);
        Tensor c = trainable("metalBackwardTraceMaxA", 3);
        Tensor d = trainable("metalBackwardTraceMaxB", 3);
        Tensor max = c.max(d);
        backendIntentPlan = backendIntentPlan.withBackend(max, ComputeBackend.GPU_METAL);
        Tensor q = trainable("metalBackwardTraceSdpaQ", 1, 2, 2);
        Tensor k = trainable("metalBackwardTraceSdpaK", 1, 2, 2);
        Tensor v = trainable("metalBackwardTraceSdpaV", 1, 2, 2);
        Tensor attention = q.scaledDotProductAttention(k, v, tensor.options.AttentionOptions.defaults().withScale(1.0));
        backendIntentPlan = backendIntentPlan.withBackend(attention, ComputeBackend.GPU_METAL);
        PreparedExecution execution = CompiledGraph.compile(
                        weightedSum(attention, "metalBackwardTraceSdpaWeight"),
                        CompileConfig.training()
                , backendIntentPlan)
                .prepare(runtimeWithRequiredAcceleratorBufferNoThreshold(ComputeBackend.GPU_METAL));
        assertFalse(execution.backwardSteps().stream()
                .anyMatch(step -> step.compiledNode().operation() != null
                        && step.compiledNode().operation().opType() == Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION_BACKWARD));
        execution.executeTraced(ExecutionMode.FORWARD_BACKWARD);
    }


    @Test
    void gpuMetalLinearBiasTanhCanExecuteThroughExplicitAppleShim() {
        String explicitLib = System.getProperty("synaptik.metal.mps.lib");
        assumeTrue(explicitLib != null && !explicitLib.isBlank());

        Tensor cpuInput = new Tensor(new float[]{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f}, new int[]{2, 3}, null, "cpuInput", DataType.FLOAT32);
        Tensor cpuWeight = new Tensor(new float[]{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f}, new int[]{3, 2}, null, "cpuWeight", DataType.FLOAT32);
        Tensor cpuBias = new Tensor(new float[]{0.1f, 0.2f}, new int[]{2}, null, "cpuBias", DataType.FLOAT32);
        Tensor cpuOut = cpuInput.linear(cpuWeight, cpuBias).tanh();
        CompiledGraph.compile(cpuOut, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        Tensor input = new Tensor(new float[]{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f}, new int[]{2, 3}, null, "input", DataType.FLOAT32);
        Tensor weight = new Tensor(new float[]{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f}, new int[]{3, 2}, null, "weight", DataType.FLOAT32);
        Tensor bias = new Tensor(new float[]{0.1f, 0.2f}, new int[]{2}, null, "bias", DataType.FLOAT32);
        Tensor linear = input.linear(weight, bias);
        Tensor out = linear.tanh();

        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();

        backendIntentPlan = backendIntentPlan.withBackend(linear, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline(), backendIntentPlan)
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .toList();
        assertEquals(1, gpuSteps.size());
        PreparedMetalExecutable executable = (PreparedMetalExecutable) testsupport.MetadataArtifacts.acceleratorExecutable(gpuSteps.getFirst().metadata());
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
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(matmul1, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(add1, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(out1, ComputeBackend.GPU_METAL);
        PreparedExecution execution1 = CompiledGraph.compile(out1, CompileConfig.noGraphOptimizationBaseline(), backendIntentPlan)
                .prepare(RuntimeConfig.inferenceDefaults());
        PreparedMetalExecutable executable1 = (PreparedMetalExecutable) execution1.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .map(step -> testsupport.MetadataArtifacts.acceleratorExecutable(step.metadata()))
                .findFirst()
                .orElseThrow();

        Tensor a2 = new Tensor(new float[]{2f, 3f, 4f, 5f, 6f, 7f}, new int[]{2, 3}, null, "a2", DataType.FLOAT32);
        Tensor b2 = new Tensor(new float[]{2f, 3f, 4f, 5f, 6f, 7f}, new int[]{3, 2}, null, "b2", DataType.FLOAT32);
        Tensor bias2 = new Tensor(new float[]{2f, -2f}, new int[]{2}, null, "bias2", DataType.FLOAT32);
        Tensor matmul2 = a2.matmul(b2);
        Tensor add2 = matmul2.add(bias2);
        Tensor out2 = add2.relu();
        backendIntentPlan = backendIntentPlan.withBackend(matmul2, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(add2, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(out2, ComputeBackend.GPU_METAL);
        PreparedExecution execution2 = CompiledGraph.compile(out2, CompileConfig.noGraphOptimizationBaseline(), backendIntentPlan)
                .prepare(RuntimeConfig.inferenceDefaults());
        PreparedMetalExecutable executable2 = (PreparedMetalExecutable) execution2.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .map(step -> testsupport.MetadataArtifacts.acceleratorExecutable(step.metadata()))
                .findFirst()
                .orElseThrow();

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
        CompiledGraph.compile(cpuOut, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{3, 2}, null, "b", DataType.FLOAT32);
        Tensor bias = new Tensor(new float[]{1f, -1f}, new int[]{2}, null, "bias", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor add = matmul.add(bias);
        Tensor out = add.relu();

        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();

        backendIntentPlan = backendIntentPlan.withBackend(matmul, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(add, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline(), backendIntentPlan)
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .toList();
        assertEquals(1, gpuSteps.size());
        PreparedMetalExecutable executable = (PreparedMetalExecutable) testsupport.MetadataArtifacts.acceleratorExecutable(gpuSteps.getFirst().metadata());
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

        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();

        backendIntentPlan = backendIntentPlan.withBackend(matmul, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(add, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline(), backendIntentPlan)
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .toList();
        assertEquals(1, gpuSteps.size());
        PreparedMetalExecutable executable = (PreparedMetalExecutable) testsupport.MetadataArtifacts.acceleratorExecutable(gpuSteps.getFirst().metadata());
        assertNotNull(executable.bridgeExecutable());
    }

    @Test
    void gpuMetalPartitionPrepareBuildsSingleRegionStepForMatmulNegChain() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{3, 2}, null, "b", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor out = matmul.neg();

        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();

        backendIntentPlan = backendIntentPlan.withBackend(matmul, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline(), backendIntentPlan)
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .toList();
        assertEquals(1, gpuSteps.size());
        assertTrue(gpuSteps.getFirst().orderedNodeIds().size() > 1);
        assertEquals(gpuSteps.getFirst().compiledNode().id(), gpuSteps.getFirst().orderedNodeIds().getLast());
        assertEquals(List.of(gpuSteps.getFirst().compiledNode().id()), gpuSteps.getFirst().boundaryOutputNodeIds());
    }

    @Test
    void gpuMetalPartitionPrepareBuildsSingleAnchorStepForMatmulReluSqrtInvChain() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{3, 2}, null, "b", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor relu = matmul.relu();
        Tensor sqrt = relu.sqrt();
        Tensor out = sqrt.inv();

        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();

        backendIntentPlan = backendIntentPlan.withBackend(matmul, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(relu, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(sqrt, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline(), backendIntentPlan)
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .toList();
        assertEquals(1, gpuSteps.size());
        PreparedMetalExecutable executable = (PreparedMetalExecutable) testsupport.MetadataArtifacts.acceleratorExecutable(gpuSteps.getFirst().metadata());
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

        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();

        backendIntentPlan = backendIntentPlan.withBackend(matmul, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(mul, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(div, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline(), backendIntentPlan)
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .toList();
        assertEquals(1, gpuSteps.size());
        PreparedMetalExecutable executable = (PreparedMetalExecutable) testsupport.MetadataArtifacts.acceleratorExecutable(gpuSteps.getFirst().metadata());
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

        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();

        backendIntentPlan = backendIntentPlan.withBackend(matmul, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(sub, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline(), backendIntentPlan)
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .toList();
        assertEquals(1, gpuSteps.size());
        PreparedMetalExecutable executable = (PreparedMetalExecutable) testsupport.MetadataArtifacts.acceleratorExecutable(gpuSteps.getFirst().metadata());
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

        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();

        backendIntentPlan = backendIntentPlan.withBackend(matmul, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(clampMin, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(clampMax, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline(), backendIntentPlan)
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .toList();
        assertEquals(1, gpuSteps.size());
        PreparedMetalExecutable executable = (PreparedMetalExecutable) testsupport.MetadataArtifacts.acceleratorExecutable(gpuSteps.getFirst().metadata());
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

        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();

        backendIntentPlan = backendIntentPlan.withBackend(matmul, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(biased, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(added, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline(), backendIntentPlan)
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .toList();
        assertEquals(1, gpuSteps.size());
        PreparedMetalExecutable executable = (PreparedMetalExecutable) testsupport.MetadataArtifacts.acceleratorExecutable(gpuSteps.getFirst().metadata());
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

        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();

        backendIntentPlan = backendIntentPlan.withBackend(matmul, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(relu, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(abs, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(add, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline(), backendIntentPlan)
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .toList();
        assertEquals(1, gpuSteps.size());
        PreparedMetalExecutable executable = (PreparedMetalExecutable) testsupport.MetadataArtifacts.acceleratorExecutable(gpuSteps.getFirst().metadata());
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

        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();

        backendIntentPlan = backendIntentPlan.withBackend(matmul, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(relu, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(abs, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(neg, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(add1, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(add2, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline(), backendIntentPlan)
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .toList();
        assertEquals(1, gpuSteps.size());
        PreparedMetalExecutable executable = (PreparedMetalExecutable) testsupport.MetadataArtifacts.acceleratorExecutable(gpuSteps.getFirst().metadata());
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

        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();

        backendIntentPlan = backendIntentPlan.withBackend(matmul, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(relu, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(abs, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(add, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        CompileConfig optimizer = CompileConfig.noGraphOptimizationBaseline()
                .withBackendPlanning(CompileConfig.noGraphOptimizationBaseline().backendPlanning()
                        .withOwnershipPlanner(config.compile.RegionOwnershipPlannerStrategy.SCORED)
                        .withSearch(new config.compile.PartitionSearchConfig(
                                1,
                                4,
                                new config.compile.PartitionScoreWeights(1000.0, 120.0, 450.0, 80.0, 60.0, 1.0)
                        )));
        PreparedExecution execution = CompiledGraph.compile(out, optimizer, backendIntentPlan)
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .toList();
        assertEquals(1, gpuSteps.size());
        PreparedMetalExecutable executable = (PreparedMetalExecutable) testsupport.MetadataArtifacts.acceleratorExecutable(gpuSteps.getFirst().metadata());
        assertEquals(1, executable.plan().lowering().dagSpec().nodes().size());
    }

    @Test
    void gpuMetalPartitionPrepareBuildsSingleAnchorStepForReshapeDag() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "b", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor reshape = matmul.reshape(1, 4);
        Tensor out = reshape.tanh();

        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();

        backendIntentPlan = backendIntentPlan.withBackend(matmul, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(reshape, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline(), backendIntentPlan)
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .toList();
        assertEquals(1, gpuSteps.size());
        PreparedMetalExecutable executable = (PreparedMetalExecutable) testsupport.MetadataArtifacts.acceleratorExecutable(gpuSteps.getFirst().metadata());
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

        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();

        backendIntentPlan = backendIntentPlan.withBackend(matmul, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(reshape, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(contiguous, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline(), backendIntentPlan)
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .toList();
        assertEquals(1, gpuSteps.size());
        PreparedMetalExecutable executable = (PreparedMetalExecutable) testsupport.MetadataArtifacts.acceleratorExecutable(gpuSteps.getFirst().metadata());
        assertEquals(4, executable.plan().lowering().dagSpec().nodes().size());
    }

    @Test
    void gpuMetalPartitionPrepareBuildsSingleAnchorStepForPermuteDag() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "b", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor permute = matmul.permute(1, 0);
        Tensor out = permute.neg();

        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();

        backendIntentPlan = backendIntentPlan.withBackend(matmul, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(permute, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline(), backendIntentPlan)
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .toList();
        assertEquals(1, gpuSteps.size());
        PreparedMetalExecutable executable = (PreparedMetalExecutable) testsupport.MetadataArtifacts.acceleratorExecutable(gpuSteps.getFirst().metadata());
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

        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();

        backendIntentPlan = backendIntentPlan.withBackend(matmul, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(reshape, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(expand, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(squeeze, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline(), backendIntentPlan)
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .toList();
        assertEquals(1, gpuSteps.size());
        PreparedMetalExecutable executable = (PreparedMetalExecutable) testsupport.MetadataArtifacts.acceleratorExecutable(gpuSteps.getFirst().metadata());
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

        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();

        backendIntentPlan = backendIntentPlan.withBackend(kPermuted, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(scores, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline(), backendIntentPlan)
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .toList();
        assertEquals(1, gpuSteps.size());
        PreparedMetalExecutable executable = (PreparedMetalExecutable) testsupport.MetadataArtifacts.acceleratorExecutable(gpuSteps.getFirst().metadata());
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

        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();

        backendIntentPlan = backendIntentPlan.withBackend(matmul, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(kPermuted, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(scores, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline(), backendIntentPlan)
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .toList();
        assertEquals(1, gpuSteps.size());
        PreparedMetalExecutable executable = (PreparedMetalExecutable) testsupport.MetadataArtifacts.acceleratorExecutable(gpuSteps.getFirst().metadata());
        assertEquals(4, executable.plan().lowering().dagSpec().nodes().size());
        assertEquals(4, executable.plan().lowering().dagSpec().externalInputs().size());
    }

    @Test
    void gpuMetalWhereCanKeepComparePredicateInsideGpuRegion() {
        Tensor left = new Tensor(new float[]{1f, 3f, 2f, 4f}, new int[]{2, 2}, null, "phase27CompareLeft", DataType.FLOAT32);
        Tensor right = new Tensor(new float[]{2f, 2f, 2f, 2f}, new int[]{2, 2}, null, "phase27CompareRight", DataType.FLOAT32);
        Tensor trueBranch = new Tensor(new float[]{10f, 20f, 30f, 40f}, new int[]{2, 2}, null, "phase27True", DataType.FLOAT32);
        Tensor falseBranch = new Tensor(new float[]{-10f, -20f, -30f, -40f}, new int[]{2, 2}, null, "phase27False", DataType.FLOAT32);
        Tensor compare = left.greaterThan(right);
        Tensor selected = Tensor.where(compare, trueBranch, falseBranch);
        Tensor out = selected.relu();

        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();

        backendIntentPlan = backendIntentPlan.withBackend(compare, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(selected, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        CompiledGraph compiled = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline(), backendIntentPlan);
        PreparedExecution execution = compiled.prepare(RuntimeConfig.inferenceDefaults());
        PartitionPlanningContext planningContext = planningContext(compiled);
        int compareNodeId = nodeId(compiled, Operation.OpType.GT);

        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .toList();
        assertEquals(1, gpuSteps.size());
        PreparedMetalExecutable executable = (PreparedMetalExecutable) testsupport.MetadataArtifacts.acceleratorExecutable(gpuSteps.getFirst().metadata());
        var manifest = executable.gpuLoweredRegionManifest();

        assertTrue(executable.plan().lowering().dagSpec().externalInputs().stream()
                .noneMatch(input -> input.dataType() == DataType.BOOL));
        assertTrue(executable.plan().lowering().dagSpec().nodes().stream()
                .anyMatch(node -> node.type() == backend.accelerator.dag.AcceleratorDagNodeType.GT));
        assertTrue(executable.plan().lowering().dagSpec().nodes().stream()
                .anyMatch(node -> node.type() == backend.accelerator.dag.AcceleratorDagNodeType.WHERE));
        assertTrue(executable.plan().lowering().dagSpec().nodes().stream()
                .anyMatch(node -> node.type() == backend.accelerator.dag.AcceleratorDagNodeType.RELU));
        assertTrue(executable.plan().lowering().dagSpec().nodes().stream()
                .anyMatch(node -> node.nodeId() == compareNodeId));
        String dtypeResidencyEvidence = manifest.backendExtensions().values().toString();
        assertTrue(dtypeResidencyEvidence.contains("role=compute dtype=BOOL"));
        assertTrue(dtypeResidencyEvidence.contains("role=internalValue dtype=BOOL residentRepresentable=true"));
        assertTrue(manifest.rejections().stream().noneMatch(rejection ->
                rejection.originalNodeId() == compareNodeId
                        && rejection.level().equals("dtype_residency.internalValue")));
        assertEquals("", MetalPartitionSupport.plannerUnsupportedReason(compiledNode(compiled, compareNodeId), planningContext));
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

        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();

        backendIntentPlan = backendIntentPlan.withBackend(matmul, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(kPermuted, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(scores, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(masked, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline(), backendIntentPlan)
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .toList();
        assertFalse(gpuSteps.isEmpty());
        int loweredNodeCount = gpuSteps.stream()
                .map(step -> (PreparedMetalExecutable) testsupport.MetadataArtifacts.acceleratorExecutable(step.metadata()))
                .mapToInt(executable -> executable.plan().lowering().dagSpec().nodes().size())
                .sum();
        assertTrue(loweredNodeCount >= 5);
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

        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();

        backendIntentPlan = backendIntentPlan.withBackend(matmul, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(kPermuted, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(scores, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(masked, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(weights, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline(), backendIntentPlan)
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .toList();
        assertFalse(gpuSteps.isEmpty());
        List<backend.accelerator.dag.AcceleratorDagNode> loweredNodes = gpuSteps.stream()
                .map(step -> (PreparedMetalExecutable) testsupport.MetadataArtifacts.acceleratorExecutable(step.metadata()))
                .flatMap(executable -> executable.plan().lowering().dagSpec().nodes().stream())
                .toList();
        assertTrue(loweredNodes.size() >= 6);
        assertTrue(loweredNodes.stream()
                .anyMatch(node -> node.type() == backend.accelerator.dag.AcceleratorDagNodeType.WHERE));
        assertTrue(loweredNodes.stream()
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

        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();

        backendIntentPlan = backendIntentPlan.withBackend(matmul, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(add, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        RuntimeConfig runtime = RuntimeConfig.inferenceDefaults().withAccelerator(
                RuntimeConfig.inferenceDefaults().accelerator().withMetal(
                        RuntimeConfig.inferenceDefaults().accelerator().metal().withEnabled(false)
                )
        );
        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline(), backendIntentPlan)
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

        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();

        backendIntentPlan = backendIntentPlan.withBackend(matmul, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(add, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        RuntimeConfig runtime = RuntimeConfig.inferenceDefaults().withAccelerator(
                RuntimeConfig.inferenceDefaults().accelerator().withMetal(
                        RuntimeConfig.inferenceDefaults().accelerator().metal().withMinimumEstimatedWork(Long.MAX_VALUE)
                )
        );
        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline(), backendIntentPlan)
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
                        CompileConfig.inference().withBackendPlanning(config.compile.BackendPlanningConfig.autoAccelerator())
                )
                .prepare(RuntimeConfig.inferenceDefaults());

        var selectedDecision = execution.prepareTrace().backendSelection().decisions().stream()
                .filter(decision -> decision.selected()
                        && ComputeBackend.GPU_METAL.name().equals(decision.selectedBackend()))
                .findFirst()
                .orElseThrow();

        assertNotNull(selectedDecision.costSummary());
        assertEquals("PROFILE_DERIVED", selectedDecision.costSummary().preset());
    }

    @Test
    void minimumWorkRejectionStillWinsOverProfileDerivedCost() {
        planning.partition.PartitionPlan plan = new planning.partition.PartitionPlan() {
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
        planning.partition.PartitionPlan tinyBoundaryHeavyPlan = new planning.partition.PartitionPlan() {
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

        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.inference())
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
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.inference(), backendIntentPlan)
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
        CompiledGraph.compile(cpuOut, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{3, 2}, null, "b", DataType.FLOAT32);
        Tensor bias = new Tensor(new float[]{1f, -1f}, new int[]{2}, null, "bias", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor add = matmul.add(bias);
        Tensor out = add.tanh();

        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();

        backendIntentPlan = backendIntentPlan.withBackend(matmul, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(add, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline(), backendIntentPlan)
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .toList();
        assertEquals(1, gpuSteps.size());
        PreparedMetalExecutable executable = (PreparedMetalExecutable) testsupport.MetadataArtifacts.acceleratorExecutable(gpuSteps.getFirst().metadata());
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
        CompiledGraph.compile(cpuOut, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{3, 2}, null, "b", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor neg = matmul.neg();
        Tensor abs = neg.abs();
        Tensor sqrt = abs.sqrt();
        Tensor out = sqrt.inv();

        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();

        backendIntentPlan = backendIntentPlan.withBackend(matmul, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(neg, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(abs, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(sqrt, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline(), backendIntentPlan)
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuStep = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .findFirst()
                .orElseThrow();
        PreparedMetalExecutable executable = (PreparedMetalExecutable) testsupport.MetadataArtifacts.acceleratorExecutable(gpuStep.metadata());
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
        CompiledGraph.compile(cpuOut, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{3, 2}, null, "b", DataType.FLOAT32);
        Tensor scale = new Tensor(new float[]{0.5f, 1.5f}, new int[]{2}, null, "scale", DataType.FLOAT32);
        Tensor denom = new Tensor(new float[]{2.0f, 4.0f}, new int[]{2}, null, "denom", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor mul = matmul.mul(scale);
        Tensor div = mul.div(denom);
        Tensor out = div.tanh();

        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();

        backendIntentPlan = backendIntentPlan.withBackend(matmul, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(mul, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(div, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline(), backendIntentPlan)
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuStep = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .findFirst()
                .orElseThrow();
        PreparedMetalExecutable executable = (PreparedMetalExecutable) testsupport.MetadataArtifacts.acceleratorExecutable(gpuStep.metadata());
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
        CompiledGraph.compile(cpuOut, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{3, 2}, null, "b", DataType.FLOAT32);
        Tensor shift = new Tensor(new float[]{0.5f, 1.5f}, new int[]{2}, null, "shift", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor sub = matmul.sub(shift);
        Tensor out = sub.tanh();

        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();

        backendIntentPlan = backendIntentPlan.withBackend(matmul, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(sub, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline(), backendIntentPlan)
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuStep = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .findFirst()
                .orElseThrow();
        PreparedMetalExecutable executable = (PreparedMetalExecutable) testsupport.MetadataArtifacts.acceleratorExecutable(gpuStep.metadata());
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
        CompiledGraph.compile(cpuOut, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{3, 2}, null, "b", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor clampMin = matmul.clampMin(0.25);
        Tensor clampMax = clampMin.clampMax(5.0);
        Tensor out = clampMax.tanh();

        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();

        backendIntentPlan = backendIntentPlan.withBackend(matmul, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(clampMin, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(clampMax, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline(), backendIntentPlan)
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuStep = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .findFirst()
                .orElseThrow();
        PreparedMetalExecutable executable = (PreparedMetalExecutable) testsupport.MetadataArtifacts.acceleratorExecutable(gpuStep.metadata());
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
        CompiledGraph.compile(cpuOut, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{3, 2}, null, "b", DataType.FLOAT32);
        Tensor bias = new Tensor(new float[]{0.5f, 1.5f}, new int[]{2}, null, "bias", DataType.FLOAT32);
        Tensor residual = new Tensor(new float[]{0.25f, 0.75f}, new int[]{2}, null, "residual", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor biased = matmul.add(bias);
        Tensor added = biased.add(residual);
        Tensor out = added.tanh();

        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();

        backendIntentPlan = backendIntentPlan.withBackend(matmul, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(biased, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(added, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline(), backendIntentPlan)
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuStep = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .findFirst()
                .orElseThrow();
        PreparedMetalExecutable executable = (PreparedMetalExecutable) testsupport.MetadataArtifacts.acceleratorExecutable(gpuStep.metadata());
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
        CompiledGraph.compile(cpuOut, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{3, 2}, null, "b", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor relu = matmul.relu();
        Tensor abs = matmul.abs();
        Tensor add = relu.add(abs);
        Tensor out = add.tanh();

        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();

        backendIntentPlan = backendIntentPlan.withBackend(matmul, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(relu, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(abs, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(add, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline(), backendIntentPlan)
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuStep = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .findFirst()
                .orElseThrow();
        PreparedMetalExecutable executable = (PreparedMetalExecutable) testsupport.MetadataArtifacts.acceleratorExecutable(gpuStep.metadata());
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
        CompiledGraph.compile(cpuOut, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{3, 2}, null, "b", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor relu = matmul.relu();
        Tensor abs = matmul.abs();
        Tensor neg = matmul.neg();
        Tensor add1 = relu.add(abs);
        Tensor add2 = add1.add(neg);
        Tensor out = add2.tanh();

        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();

        backendIntentPlan = backendIntentPlan.withBackend(matmul, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(relu, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(abs, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(neg, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(add1, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(add2, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline(), backendIntentPlan)
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuStep = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .findFirst()
                .orElseThrow();
        PreparedMetalExecutable executable = (PreparedMetalExecutable) testsupport.MetadataArtifacts.acceleratorExecutable(gpuStep.metadata());
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
        CompiledGraph.compile(cpuOut, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "b", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor reshape = matmul.reshape(1, 4);
        Tensor out = reshape.tanh();

        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();

        backendIntentPlan = backendIntentPlan.withBackend(matmul, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(reshape, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline(), backendIntentPlan)
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuStep = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .findFirst()
                .orElseThrow();
        PreparedMetalExecutable executable = (PreparedMetalExecutable) testsupport.MetadataArtifacts.acceleratorExecutable(gpuStep.metadata());
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
        CompiledGraph.compile(cpuOut, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "b", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor permute = matmul.permute(1, 0);
        Tensor out = permute.neg();

        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();

        backendIntentPlan = backendIntentPlan.withBackend(matmul, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(permute, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline(), backendIntentPlan)
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuStep = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .findFirst()
                .orElseThrow();
        PreparedMetalExecutable executable = (PreparedMetalExecutable) testsupport.MetadataArtifacts.acceleratorExecutable(gpuStep.metadata());
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
        CompiledGraph.compile(cpuOut, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "b", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor reshape = matmul.reshape(1, 4);
        Tensor expand = reshape.expandDims(0);
        Tensor squeeze = expand.squeeze(0);
        Tensor out = squeeze.tanh();

        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();

        backendIntentPlan = backendIntentPlan.withBackend(matmul, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(reshape, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(expand, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(squeeze, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline(), backendIntentPlan)
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuStep = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .findFirst()
                .orElseThrow();
        PreparedMetalExecutable executable = (PreparedMetalExecutable) testsupport.MetadataArtifacts.acceleratorExecutable(gpuStep.metadata());
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
        CompiledGraph.compile(cpuOut, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

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

        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();

        backendIntentPlan = backendIntentPlan.withBackend(kPermuted, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(scores, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline(), backendIntentPlan)
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuStep = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .findFirst()
                .orElseThrow();
        PreparedMetalExecutable executable = (PreparedMetalExecutable) testsupport.MetadataArtifacts.acceleratorExecutable(gpuStep.metadata());
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
        CompiledGraph.compile(cpuOut, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

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

        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();

        backendIntentPlan = backendIntentPlan.withBackend(matmul, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(kPermuted, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(scores, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline(), backendIntentPlan)
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuStep = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .findFirst()
                .orElseThrow();
        PreparedMetalExecutable executable = (PreparedMetalExecutable) testsupport.MetadataArtifacts.acceleratorExecutable(gpuStep.metadata());
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
        CompiledGraph.compile(cpuOut, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

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

        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();

        backendIntentPlan = backendIntentPlan.withBackend(matmul, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(kPermuted, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(scores, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(masked, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline(), backendIntentPlan)
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuStep = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .findFirst()
                .orElseThrow();
        PreparedMetalExecutable executable = (PreparedMetalExecutable) testsupport.MetadataArtifacts.acceleratorExecutable(gpuStep.metadata());
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
        CompiledGraph.compile(cpuOut, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

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

        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();

        backendIntentPlan = backendIntentPlan.withBackend(matmul, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(kPermuted, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(scores, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(masked, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(weights, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline(), backendIntentPlan)
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuStep = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .findFirst()
                .orElseThrow();
        PreparedMetalExecutable executable = (PreparedMetalExecutable) testsupport.MetadataArtifacts.acceleratorExecutable(gpuStep.metadata());
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
        CompiledGraph.compile(cpuOut, CompileConfig.noGraphOptimizationBaseline())
                .prepare(RuntimeConfig.inferenceDefaults()).execute(ExecutionMode.FORWARD);

        Tensor a = new Tensor(new float[]{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f}, new int[]{2, 3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f}, new int[]{3, 2}, null, "b", DataType.FLOAT32);
        Tensor bias = new Tensor(new float[]{0.1f, 0.2f}, new int[]{2}, null, "bias", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor add = matmul.add(bias);
        Tensor out = add.exp();

        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();

        backendIntentPlan = backendIntentPlan.withBackend(matmul, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(add, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline(), backendIntentPlan)
                .prepare(RuntimeConfig.inferenceDefaults());

        var gpuSteps = execution.forwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .toList();
        assertEquals(1, gpuSteps.size());
        PreparedMetalExecutable executable = (PreparedMetalExecutable) testsupport.MetadataArtifacts.acceleratorExecutable(gpuSteps.getFirst().metadata());
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

        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.inference())
                .prepare(bfloat16BlasRuntime());

        var linearStep = execution.forwardSteps().stream()
                .filter(step -> step.compiledNode().operation() != null && step.compiledNode().operation().opType() == Operation.OpType.LINEAR)
                .findFirst()
                .orElseThrow();
        var logSoftmaxStep = execution.forwardSteps().stream()
                .filter(step -> hasLabel(step, "logSoftmax"))
                .findFirst()
                .orElseThrow();
        Cpu1PreparedArtifact artifact = assertInstanceOf(Cpu1PreparedArtifact.class, linearStep.metadata().artifact());
        Cpu1PreparedMatmulUnit preparedMatmulUnit = artifact.preparedMatmulUnit();

        assertEquals(Operation.OpType.LINEAR, linearStep.compiledNode().operation().opType());
        assertEquals(1, linearStep.orderedNodeIds().size());
        assertEquals(DataType.BFLOAT16, preparedMatmulUnit.dataType());
        assertEquals(Cpu1StorageKind.JAVA_ARRAY, preparedMatmulUnit.storageKind());
        assertEquals(Cpu1MatmulRoute.JAVA_SCALAR, preparedMatmulUnit.route());
        assertEquals(Cpu1MatmulKernelId.MATMUL_BF16_DENSE_SCALAR, preparedMatmulUnit.kernelId());
        assertEquals(Cpu1MatmulPostOp.ADD_BIAS, preparedMatmulUnit.postOp());
        assertNotNull(testsupport.MetadataArtifacts.cpuPlan(logSoftmaxStep.metadata()));
        assertFalse(logSoftmaxStep.compiledNode().operation() != null
                && logSoftmaxStep.compiledNode().operation().opType() == Operation.OpType.LOG_SOFTMAX);
    }

    @Test
    void bfloat16SoftmaxToMeanUsesCpu1LinearSpecializationAndPreparedReduction() {
        Tensor input = new Tensor(new double[32 * 64], new int[]{32, 64}, null, "input", DataType.BFLOAT16);
        Tensor weight = new Tensor(new double[64 * 96], new int[]{64, 96}, null, "weight", DataType.BFLOAT16);
        Tensor bias = new Tensor(new double[96], new int[]{96}, null, "bias", DataType.BFLOAT16);
        Tensor out = input.linear(weight, bias).softmax(1).mean(1);

        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.inference())
                .prepare(bfloat16BlasRuntime());

        var linearStep = execution.forwardSteps().stream()
                .filter(step -> step.compiledNode().operation() != null && step.compiledNode().operation().opType() == Operation.OpType.LINEAR)
                .findFirst()
                .orElseThrow();
        var softmaxStep = execution.forwardSteps().stream()
                .filter(step -> hasLabel(step, "softmax"))
                .findFirst()
                .orElseThrow();
        Cpu1PreparedArtifact artifact = assertInstanceOf(Cpu1PreparedArtifact.class, linearStep.metadata().artifact());
        Cpu1PreparedMatmulUnit preparedMatmulUnit = artifact.preparedMatmulUnit();

        assertEquals(Operation.OpType.LINEAR, linearStep.compiledNode().operation().opType());
        assertEquals(1, linearStep.orderedNodeIds().size());
        assertEquals(DataType.BFLOAT16, preparedMatmulUnit.dataType());
        assertEquals(Cpu1StorageKind.JAVA_ARRAY, preparedMatmulUnit.storageKind());
        assertEquals(Cpu1MatmulRoute.JAVA_SCALAR, preparedMatmulUnit.route());
        assertEquals(Cpu1MatmulKernelId.MATMUL_BF16_DENSE_SCALAR, preparedMatmulUnit.kernelId());
        assertEquals(Cpu1MatmulPostOp.ADD_BIAS, preparedMatmulUnit.postOp());
        assertNotNull(testsupport.MetadataArtifacts.cpuPlan(softmaxStep.metadata()));
        assertFalse(softmaxStep.compiledNode().operation() != null
                && softmaxStep.compiledNode().operation().opType() == Operation.OpType.SOFTMAX);
    }

    @Test
    void bfloat16LayerNormToMeanKeepsFloatContinuationInInference() {
        Tensor input = new Tensor(new double[32 * 64], new int[]{32, 64}, null, "input", DataType.BFLOAT16);
        Tensor gamma = new Tensor(new double[64], new int[]{64}, null, "gamma", DataType.BFLOAT16);
        Tensor beta = new Tensor(new double[64], new int[]{64}, null, "beta", DataType.BFLOAT16);
        Tensor out = input.layerNorm(gamma, beta, 1e-5).mean(1);

        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.inference())
                .prepare(bfloat16BlasRuntime());

        var layerNormStep = execution.forwardSteps().stream()
                .filter(step -> step.compiledNode().operation() != null && step.compiledNode().operation().opType() == Operation.OpType.LAYER_NORM)
                .findFirst()
                .orElseThrow();

        assertTrue(testsupport.MetadataArtifacts.cpuPlan(layerNormStep.metadata()).publishFloatContinuation());
        assertEquals("BFLOAT16", testsupport.MetadataArtifacts.cpuPlan(layerNormStep.metadata()).computeContract().storageType().name());
        assertEquals("F32", testsupport.MetadataArtifacts.cpuPlan(layerNormStep.metadata()).computeContract().computeType().name());
    }

    @Test
    void bfloat16RmsNormToMeanKeepsFloatContinuationInInference() {
        Tensor input = new Tensor(new double[32 * 64], new int[]{32, 64}, null, "input", DataType.BFLOAT16);
        Tensor gamma = new Tensor(new double[64], new int[]{64}, null, "gamma", DataType.BFLOAT16);
        Tensor out = input.rmsNorm(gamma, 1e-5).mean(1);

        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.inference())
                .prepare(bfloat16BlasRuntime());

        var rmsNormStep = execution.forwardSteps().stream()
                .filter(step -> step.compiledNode().operation() != null && step.compiledNode().operation().opType() == Operation.OpType.RMS_NORM)
                .findFirst()
                .orElseThrow();

        assertTrue(testsupport.MetadataArtifacts.cpuPlan(rmsNormStep.metadata()).publishFloatContinuation());
        assertEquals("BFLOAT16", testsupport.MetadataArtifacts.cpuPlan(rmsNormStep.metadata()).computeContract().storageType().name());
        assertEquals("F32", testsupport.MetadataArtifacts.cpuPlan(rmsNormStep.metadata()).computeContract().computeType().name());
    }

    @Test
    void bfloat16LogSoftmaxToNllLossKeepsFloatContinuationInInference() {
        Tensor logits = new Tensor(new double[16 * 8], new int[]{16, 8}, null, "logits", DataType.BFLOAT16);
        Tensor targets = new Tensor(new double[16 * 8], new int[]{16, 8}, null, "targets", DataType.BFLOAT16);
        Tensor out = logits.logSoftmax(1).nllLoss(targets, 1);

        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.inference())
                .prepare(bfloat16BlasRuntime());

        var logSoftmaxStep = execution.forwardSteps().stream()
                .filter(step -> hasLabel(step, "logSoftmax"))
                .findFirst()
                .orElseThrow();

        assertNotNull(testsupport.MetadataArtifacts.cpuPlan(logSoftmaxStep.metadata()));
        assertFalse(logSoftmaxStep.compiledNode().operation() != null
                && logSoftmaxStep.compiledNode().operation().opType() == Operation.OpType.LOG_SOFTMAX);
    }

    @Test
    void bfloat16MatmulToCrossEntropyLossPublishesFloatContinuationInInference() {
        Tensor a = new Tensor(new double[32 * 64], new int[]{32, 64}, null, "a", DataType.BFLOAT16);
        Tensor b = new Tensor(new double[64 * 96], new int[]{64, 96}, null, "b", DataType.BFLOAT16);
        Tensor targets = new Tensor(new double[32 * 96], new int[]{32, 96}, null, "targets", DataType.BFLOAT16);
        Tensor out = a.matmul(b).crossEntropyLoss(targets, 1);

        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.inference())
                .prepare(bfloat16BlasRuntime());

        var matmulStep = execution.forwardSteps().stream()
                .filter(step -> step.compiledNode().operation() != null && step.compiledNode().operation().opType() == Operation.OpType.MATMUL)
                .findFirst()
                .orElseThrow();

        assertTrue(testsupport.MetadataArtifacts.cpuPlan(matmulStep.metadata()).publishFloatContinuation());
    }

    @Test
    void bfloat16SoftmaxGradPublishesFloatContinuationInTrainingUnaryChain() {
        Tensor input = new Tensor(new double[16 * 8], new int[]{16, 8}, null, "input", DataType.BFLOAT16);
        input.setRequiresGrad(true);
        Tensor out = input.exp().softmax(1).sum();

        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.training())
                .prepare(bfloat16BlasRuntime());

        var continuationStep = execution.backwardSteps().stream()
                .filter(step -> step.compiledNode().operation() != null
                        && step.compiledNode().dataType() == DataType.BFLOAT16
                        && testsupport.MetadataArtifacts.cpuPlan(step.metadata()) != null
                        && testsupport.MetadataArtifacts.cpuPlan(step.metadata()).publishFloatContinuation())
                .findFirst()
                .orElseThrow();
        assertFalse(continuationStep.compiledNode().operation().opType() == Operation.OpType.SOFTMAX_GRAD);
    }

    @Test
    void bfloat16LogSoftmaxGradPublishesFloatContinuationInTrainingUnaryChain() {
        Tensor input = new Tensor(new double[16 * 8], new int[]{16, 8}, null, "input", DataType.BFLOAT16);
        input.setRequiresGrad(true);
        Tensor out = input.exp().logSoftmax(1).sum();

        PreparedExecution execution = CompiledGraph.compile(out, CompileConfig.training())
                .prepare(bfloat16BlasRuntime());

        var continuationStep = execution.backwardSteps().stream()
                .filter(step -> step.compiledNode().operation() != null
                        && step.compiledNode().dataType() == DataType.BFLOAT16
                        && testsupport.MetadataArtifacts.cpuPlan(step.metadata()) != null
                        && testsupport.MetadataArtifacts.cpuPlan(step.metadata()).publishFloatContinuation())
                .findFirst()
                .orElseThrow();
        assertFalse(continuationStep.compiledNode().operation().opType() == Operation.OpType.LOG_SOFTMAX_GRAD);
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
        return kernelWithVectorMinAndFusedAsmWidth(vectorMinSize, KernelTuningConfig.defaultsInference().cpu().fusedAsmVectorWidth());
    }

    private static RuntimeConfig runtimeWithFusedAsmWidth(int fusedAsmWidth) {
        return new RuntimeConfig(
                kernelWithVectorMinAndFusedAsmWidth(1, fusedAsmWidth),
                config.runtime.ApproximationConfig.defaults(),
                config.runtime.BlasConfig.disabled(),
                new FusedExecutionPolicy(true)
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
                        cpu.sumAccuracyMode(),
                        cpu.matMulParallelMinSize(),
                        cpu.attentionMatMulPolicy()
                ),
                base.cuda(),
                base.opencl()
        );
    }

    private static CompileConfig fuseOnlyInferenceConfig() {
        return CompileConfig.inference()
                .withGraphOptimization(config.compile.GraphOptimizationConfig.noGraphOptimization());
    }

    private static int nodeId(CompiledGraph compiled, Operation.OpType opType) {
        return compiled.program().compiledNodes().stream()
                .filter(node -> node.operation() != null && node.operation().opType() == opType)
                .map(graph.model.CompiledNode::id)
                .findFirst()
                .orElseThrow();
    }

    private static int nodeId(CompiledGraph compiled, String label) {
        return compiled.program().compiledNodes().stream()
                .filter(node -> label.equals(node.label()))
                .map(graph.model.CompiledNode::id)
                .findFirst()
                .orElseThrow();
    }

    private static boolean hasLabel(PreparedExecutionStep step, String label) {
        return label.equals(step.compiledNode().label());
    }

    private static int nodeId(List<CompiledNode> nodes, Operation.OpType opType) {
        return nodes.stream()
                .filter(node -> node.operation() != null && node.operation().opType() == opType)
                .map(graph.model.CompiledNode::id)
                .findFirst()
                .orElseThrow();
    }

    private static int nodeId(List<CompiledNode> nodes, String label) {
        return nodes.stream()
                .filter(node -> label.equals(node.label()))
                .map(graph.model.CompiledNode::id)
                .findFirst()
                .orElseThrow();
    }

    private static Map<Integer, List<CompiledNode>> consumerMap(List<CompiledNode> nodes) {
        Map<Integer, List<CompiledNode>> consumers = new HashMap<>();
        for (CompiledNode node : nodes) {
            consumers.computeIfAbsent(node.id(), ignored -> new ArrayList<>());
        }
        for (CompiledNode node : nodes) {
            for (int inputId : node.inputIds()) {
                consumers.computeIfAbsent(inputId, ignored -> new ArrayList<>()).add(node);
            }
        }
        return consumers;
    }

    private static graph.model.CompiledNode compiledNode(CompiledGraph compiled, int nodeId) {
        return compiled.program().compiledNodes().stream()
                .filter(node -> node.id() == nodeId)
                .findFirst()
                .orElseThrow();
    }

    private static Tensor trainable(String label, int... shape) {
        int size = 1;
        for (int dim : shape) {
            size *= dim;
        }
        float[] data = new float[size];
        for (int i = 0; i < data.length; i++) {
            data[i] = (float) (Math.sin(i * 0.17d + label.length() * 0.01d) + 0.1d * i);
        }
        Tensor tensor = new Tensor(data, shape, null, label, DataType.FLOAT32);
        tensor.setRequiresGrad(true);
        return tensor;
    }

    private static Tensor weightedSum(Tensor value, String label) {
        int[] shape = value.getShapeUnsafe();
        int size = 1;
        for (int dim : shape) {
            size *= dim;
        }
        float[] data = new float[size];
        for (int i = 0; i < data.length; i++) {
            data[i] = 0.25f + (i % 5) * 0.125f;
        }
        return value.mul(new Tensor(data, shape, null, label, DataType.FLOAT32)).sum();
    }

    private static void assertMetalBackwardBufferBinding(Tensor loss, Operation.OpType opType, int minSteps) {
        PreparedExecution execution = CompiledGraph.compile(loss, CompileConfig.training())
                .prepare(runtimeWithRequiredAcceleratorBufferNoThreshold(ComputeBackend.GPU_METAL));
        var preparedSteps = execution.backwardSteps().stream()
                .filter(step -> step.metadata().backend() == ComputeBackend.GPU_METAL)
                .filter(step -> step.compiledNode().operation() != null && step.compiledNode().operation().opType() == opType)
                .toList();
        assertTrue(preparedSteps.size() >= minSteps, opType.name());

        var trace = execution.executeTraced(ExecutionMode.FORWARD_BACKWARD);
        var tracedSteps = trace.steps().stream()
                .filter(step -> "GPU_METAL".equals(step.backend()))
                .filter(step -> opType.name().equals(step.opType()))
                .toList();
        assertTrue(tracedSteps.size() >= minSteps, opType.name());
        for (var step : tracedSteps) {
            Map<String, Object> attrs = step.metadata().attributes();
            assertEquals("BUFFER_BINDING", attrs.get("acceleratorBufferExecutionPath"), opType.name());
            assertEquals("BUFFER_BINDING_AVAILABLE", attrs.get("acceleratorBufferReasonCode"), opType.name());
            assertEquals("BUFFER_BINDING", attrs.get("metalExecutionPath"), opType.name());
            assertEquals(false, attrs.get("metalUsedCpuFallback"), opType.name());
            assertTrue(((Number) attrs.get("metalNativeExecuteNs")).longValue() > 0L, opType.name());
        }
    }

    private static void assertMetalBackwardRequiredBufferRejects(Tensor loss, String... expectedSubstrings) {
        PreparedExecution execution = CompiledGraph.compile(loss, CompileConfig.training())
                .prepare(runtimeWithRequiredAcceleratorBufferNoThreshold(ComputeBackend.GPU_METAL));

        IllegalStateException failure = assertThrows(
                IllegalStateException.class,
                () -> execution.executeTraced(ExecutionMode.FORWARD_BACKWARD)
        );
        assertContainsAll(failure.getMessage(), expectedSubstrings);
    }

    private static RuntimeConfig runtimeWithRequiredAcceleratorBufferNoThreshold(ComputeBackend backend) {
        RuntimeConfig defaults = RuntimeConfig.inferenceDefaults();
        AcceleratorBackendConfig required = defaults.accelerator().forBackend(backend).withBuffer(
                new AcceleratorBufferConfig(AcceleratorBufferBindingMode.REQUIRE, true, 0)
        );
        AcceleratorConfig accelerator = switch (backend) {
            case GPU_METAL -> defaults.accelerator().withMetal(required);
            case GPU_CUDA -> defaults.accelerator().withCuda(required);
            case GPU_OPENCL -> defaults.accelerator().withOpencl(required);
            case CPU -> defaults.accelerator();
        };
        return defaults.withAccelerator(accelerator);
    }

    private static PartitionPlanningContext planningContext(CompiledGraph compiled) {
        List<CompiledNode> nodes = compiled.program().compiledNodes();
        return new PartitionPlanningContext(
                false,
                nodes,
                compiled.program().descriptorIndex(),
                consumerMap(nodes)
        );
    }

    private static boolean hasSelectedAcceleratorDecisionFor(PreparedExecution execution, ComputeBackend backend, int nodeId) {
        return execution.prepareTrace().backendSelection().decisions().stream()
                .anyMatch(decision -> decision.selected()
                        && backend.name().equals(decision.selectedBackend())
                        && decision.nodeIds().contains(nodeId));
    }

    private static void assertCpuPreparedStepAvailable(PreparedExecution execution, int nodeId) {
        assertTrue(execution.forwardSteps().stream()
                .anyMatch(step -> step.compiledNode().id() == nodeId && step.metadata().backend() == ComputeBackend.CPU));
    }

    private static void assertAcceleratorPreparedStepAvailable(PreparedExecution execution, int nodeId) {
        assertTrue(execution.forwardSteps().stream()
                .anyMatch(step -> step.compiledNode().id() == nodeId
                        && step.metadata().backend() != ComputeBackend.CPU
                        && testsupport.MetadataArtifacts.acceleratorExecutable(step.metadata()) != null));
    }

    private static void assertContainsAll(String actual, String... expectedSubstrings) {
        for (String expected : expectedSubstrings) {
            assertTrue(actual.contains(expected), () -> "Expected '" + actual + "' to contain '" + expected + "'");
        }
    }
}
