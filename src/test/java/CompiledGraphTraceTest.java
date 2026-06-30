import planning.descriptor.CompiledTensorDescriptorBuilder;
import planning.descriptor.CompiledTensorDescriptorIndex;
import runtime.device.buffer.AcceleratorBufferDecision;
import runtime.device.buffer.AcceleratorBufferExecutionPath;
import runtime.device.buffer.AcceleratorBufferReasonCode;
import backend.accelerator.exec.PreparedAcceleratorExecutable;
import backend.accelerator.lowering.GpuCompoundPatternType;
import backend.accelerator.lowering.GpuCompoundRegionSummary;
import backend.accelerator.lowering.GpuLoweredPrimitiveManifest;
import backend.accelerator.lowering.GpuLoweredRegionCandidateSpan;
import backend.accelerator.lowering.GpuLoweredRegionManifest;
import tuning.benchmark.report.GpuLoweredRegionTraceRenderer;
import backend.accelerator.lowering.GpuLoweredRegionOriginalOp;
import backend.accelerator.lowering.GpuLoweredRegionRejection;
import backend.accelerator.lowering.GpuLoweredRegionValueAssumption;
import backend.accelerator.lowering.GpuLoweringUnsupportedReason;
import backend.cuda.lowering.CudaGpuBackendPartitionCapability;
import backend.lowering.LoweringFamily;
import backend.lowering.region.EmptyRegionPayload;
import backend.lowering.region.RegionCost;
import backend.lowering.region.RegionDecision;
import backend.lowering.region.RegionExecutionGroup;
import backend.lowering.region.RegionExecutionKind;
import backend.lowering.region.RegionExecutionPlan;
import backend.lowering.region.RegionStorageContract;
import runtime.contract.CpuMaterializationReason;
import runtime.contract.ExecutionMode;
import backend.contract.ComputeBackend;
import backend.runtime.ExecutionContext;
import config.runtime.AcceleratorBufferBindingMode;
import config.compile.CompileConfig;
import config.profile.ExecutionProfile;
import config.profile.WorkloadProfile;
import graph.model.CompiledNode;
import graph.execution.plan.CompiledNodeExecutionMetadata;
import graph.execution.PreparedExecution;
import graph.execution.PreparedExecutionStep;
import planning.partition.PartitionPlanningContext;
import planning.partition.PartitionTarget;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.internal.TensorPrimitiveBuilder;
import planning.intent.BackendIntentPlan;

import java.util.List;
import java.util.Map;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertFalse;

public class CompiledGraphTraceTest {
    @Test
    void gpuLoweringCoverageSelectionTraceNamesSupportedLogSoftmaxRegion() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "traceLogSoftmaxInput", DataType.FLOAT32);
        Tensor weight = new Tensor(new float[]{1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f}, new int[]{3, 3}, null, "traceLogSoftmaxWeight", DataType.FLOAT32);
        Tensor matmul = input.matmul(weight);
        Tensor out = specialLogSoftmax(matmul, 1);
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(matmul, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        graph.CompiledGraph compiled = graph.CompiledGraph.compile(out, CompileConfig.inference(), backendIntentPlan);
        PreparedExecution prepared = compiled.prepare(config.runtime.RuntimeConfig.inferenceDefaults());
        int matmulNodeId = nodeId(compiled, operations.Operation.OpType.MATMUL);
        int logSoftmaxNodeId = nodeId(compiled, operations.Operation.OpType.LOG_SOFTMAX);

        var selected = prepared.prepareTrace().backendSelection().decisions().stream()
                .filter(decision -> decision.selected() && ComputeBackend.GPU_METAL.name().equals(decision.selectedBackend()))
                .filter(decision -> decision.nodeIds().contains(matmulNodeId) && decision.nodeIds().contains(logSoftmaxNodeId))
                .findFirst()
                .orElseThrow();
        List<String> selectedOpNames = selected.nodeIds().stream()
                .map(nodeId -> compiledNode(compiled, nodeId).operation().opType().name())
                .toList();

        assertEquals("selected", selected.reason());
        assertTrue(selectedOpNames.contains("MATMUL") || selectedOpNames.contains("LINEAR"));
        assertTrue(selectedOpNames.contains("LOG_SOFTMAX"));
    }

    @Test
    void gpuLoweredRegionManifestTraceContainsOriginalOpsAndPrimitives() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "manifestInput", DataType.FLOAT32);
        Tensor out = specialLogSoftmax(input, 1);
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        graph.CompiledGraph compiled = graph.CompiledGraph.compile(out, CompileConfig.inference(), backendIntentPlan);
        PreparedExecution prepared = compiled.prepare(config.runtime.RuntimeConfig.inferenceDefaults());
        int logSoftmaxNodeId = nodeId(compiled, operations.Operation.OpType.LOG_SOFTMAX);

        var manifest = prepared.prepareTrace().backendSelection().decisions().stream()
                .filter(decision -> decision.selected() && ComputeBackend.GPU_METAL.name().equals(decision.selectedBackend()))
                .filter(decision -> decision.nodeIds().contains(logSoftmaxNodeId))
                .map(trace.prepare.BackendSelectionDecisionTrace::gpuLoweredRegionManifest)
                .filter(candidate -> candidate != null)
                .findFirst()
                .orElseThrow();

        assertEquals("GPU_METAL", manifest.backend());
        assertTrue(manifest.selectedRegionLength() >= 1);
        assertTrue(manifest.originalOperations().stream().anyMatch(op -> op.nodeId() == logSoftmaxNodeId));
        assertTrue(manifest.originalOperations().stream().anyMatch(op -> "LOG_SOFTMAX".equals(op.opType())));
        assertTrue(manifest.loweredPrimitives().stream().anyMatch(primitive -> "SOFTMAX".equals(primitive.primitiveType())));
        assertTrue(manifest.loweredPrimitives().stream().anyMatch(primitive -> "LOG".equals(primitive.primitiveType())));
        assertTrue(manifest.inputAssumptions().stream().anyMatch(assumption -> !assumption.layout().isBlank()));
        assertTrue(manifest.candidateSpan().acceptedNodeIds().contains(logSoftmaxNodeId));
    }

    @Test
    void gpuLoweringCoverageTraceSelectsSupportedReduction() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "traceReductionInput", DataType.FLOAT32);
        Tensor out = input.sum(1);
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_CUDA);
        graph.CompiledGraph compiled = graph.CompiledGraph.compile(out, CompileConfig.inference(), backendIntentPlan);
        PreparedExecution prepared = compiled.prepare(config.runtime.RuntimeConfig.inferenceDefaults());
        int sumNodeId = nodeId(compiled, operations.Operation.OpType.SUM);
        String reason = CudaGpuBackendPartitionCapability.plannerUnsupportedReason(compiledNode(compiled, sumNodeId), null);

        assertEquals("", reason);
        assertTrue(prepared.prepareTrace().backendSelection().decisions().stream()
                .anyMatch(decision -> decision.selected()
                        && ComputeBackend.GPU_CUDA.name().equals(decision.selectedBackend())
                        && decision.nodeIds().contains(sumNodeId)));
    }

    @Test
    void gpuCompoundTraceAdmitsSupportedNormalization() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "traceReductionAdjacentInput", DataType.FLOAT32);
        Tensor gamma = new Tensor(new float[]{1f, 1f}, new int[]{2}, null, "traceReductionAdjacentGamma", DataType.FLOAT32);
        Tensor beta = new Tensor(new float[]{0f, 0f}, new int[]{2}, null, "traceReductionAdjacentBeta", DataType.FLOAT32);
        Tensor out = input.layerNorm(gamma, beta, 1.0e-5);
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_CUDA);
        graph.CompiledGraph compiled = graph.CompiledGraph.compile(out, CompileConfig.inference(), backendIntentPlan);
        PreparedExecution prepared = compiled.prepare(config.runtime.RuntimeConfig.inferenceDefaults());
        int layerNormNodeId = nodeId(compiled, operations.Operation.OpType.LAYER_NORM);
        String reason = CudaGpuBackendPartitionCapability.plannerUnsupportedReason(
                compiledNode(compiled, layerNormNodeId),
                planningContext(compiled)
        );

        assertTrue(prepared.prepareTrace().backendSelection().decisions().stream()
                .anyMatch(decision -> decision.selected()
                        && ComputeBackend.GPU_CUDA.name().equals(decision.selectedBackend())
                        && decision.nodeIds().contains(layerNormNodeId)));
        assertEquals("", reason);
    }

    @Test
    void compiledGraphExposesCompilePrepareAndRunTrace() {
        Tensor a = new Tensor(new double[]{1, 2, 3, 4}, new int[]{4}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{5, 6, 7, 8}, new int[]{4}, null, "b", DataType.FLOAT64);
        Tensor out = a.add(b).mul(a);

        graph.CompiledGraph compiled = graph.CompiledGraph.compile(out, CompileConfig.inference());
        var runTrace = compiled.prepare(config.runtime.RuntimeConfig.inferenceDefaults()).executeTraced(ExecutionMode.FORWARD);

        assertTrue(compiled.compileTrace().measured());
        assertTrue(compiled.compileTrace().totalNodeCount() > 0);
        assertTrue(compiled.compileTrace().partitionPlanning() != null);
        assertTrue(compiled.compileTrace().optimizerTrace().costExplanations().stream()
                .anyMatch(explanation -> "GraphSimplificationCostModel".equals(explanation.modelName())));
        assertTrue(runTrace.durationNs() >= 0L);
        assertTrue(runTrace.steps().size() > 0);
        assertEquals("FORWARD", runTrace.mode().name());
    }

    @Test
    void fusedHotPathPublishesPrepareAndRunTraceMetadata() {
        int size = 4096;
        float[] av = new float[size];
        float[] bv = new float[size];
        float[] cv = new float[size];
        for (int i = 0; i < size; i++) {
            av[i] = i * 0.01f;
            bv[i] = 1.0f + i * 0.02f;
            cv[i] = -0.5f + i * 0.03f;
        }
        Tensor a = new Tensor(av, new int[]{size}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(bv, new int[]{size}, null, "b", DataType.FLOAT32);
        Tensor c = new Tensor(cv, new int[]{size}, null, "c", DataType.FLOAT32);
        Tensor out = a.add(b).mul(c).relu().exp();

        graph.CompiledGraph compiled = graph.CompiledGraph.compile(out, CompileConfig.inference());
        var prepared = compiled.prepare(config.runtime.RuntimeConfig.inferenceDefaults());
        var runTrace = prepared.executeTraced(ExecutionMode.FORWARD);

        assertTrue(prepared.prepareTrace().measured());
        assertTrue(prepared.prepareTrace().durationNs() >= 0L);
        assertTrue(runTrace.durationNs() >= 0L);
        var fusedStep = runTrace.steps().stream()
                .filter(step -> step.metadata().fused() != null)
                .findFirst()
                .orElseThrow();
        assertTrue(fusedStep.metadata().fused().fusedNodeCount() > 1);
        assertTrue(!fusedStep.metadata().fused().executionBackend().isBlank());
        assertTrue(fusedStep.metadata().fused().schedulerSignature() != null
                && !fusedStep.metadata().fused().schedulerSignature().isBlank());
    }

    @Test
    void autotuneSessionCanRunWithoutCompiledGraphBackReference() {
        Tensor out = Tensor.scalar(2.0).add(Tensor.scalar(3.0));

        ExecutionProfile profile = new ExecutionProfile(
                "delegate",
                "delegate",
                DataType.FLOAT64,
                ExecutionMode.FORWARD,
                CompileConfig.noGraphOptimizationBaseline(),
                config.runtime.RuntimeConfig.inferenceDefaults(),
                WorkloadProfile.none()
        );

        tuning.autotune.TuningResult result = tuning.autotune.AutotuneSession.create(new tuning.autotune.AutotuneRequest(
                new tuning.workload.TensorRootWorkloadSpec(
                        "delegate_workload",
                        tuning.workload.WorkloadKind.GENERIC,
                        environment -> Tensor.scalar(2.0).add(Tensor.scalar(3.0)),
                        environment -> tuning.validate.ValidationReference.snapshot(
                                tuning.validate.TensorSnapshot.capture("out", Tensor.scalar(5.0)),
                                java.util.Map.of(),
                                java.util.List.of()
                        ),
                        environment -> tuning.workload.WorkloadMetadata.of("delegate_workload", tuning.workload.WorkloadKind.GENERIC)
                ),
                new tuning.candidate.ListCandidateSpace(java.util.List.of(new tuning.candidate.Candidate("delegate", profile))),
                new tuning.measure.MeasurementPolicy(0, 1, 1, true, true, true, true, false),
                tuning.validate.ValidationPolicy.defaults(),
                new tuning.search.SearchPolicy(4, 1, 1, false),
                tuning.store.PersistencePolicy.disabled()
        )).run();

        assertTrue(result.bestProfile() != null);
    }

    @Test
    void applePartitionTraceCapturesLargestStructuralDagCandidateWhenLowererRejectsIt() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{3, 2}, null, "b", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor relu = matmul.relu();
        Tensor abs = matmul.abs();
        Tensor out = relu.add(abs);

        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();

        backendIntentPlan = backendIntentPlan.withBackend(matmul, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(relu, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(abs, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        graph.CompiledGraph compiled = graph.CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline(), backendIntentPlan);
        var decisions = compiled.compileTrace().partitionPlanning().decisions();

        assertTrue(decisions.stream().anyMatch(decision ->
                decision.structuralNodeIds().size() >= 4
                        && decision.opTypes().contains("RELU")
                        && decision.opTypes().contains("ABS")
                        && decision.opTypes().contains("ADD")
        ));
        assertTrue(decisions.stream().allMatch(decision -> decision.exploredCandidates() >= 0));
        assertTrue(decisions.stream().allMatch(decision -> !decision.searchBudgetHit()));
        assertTrue(decisions.stream().allMatch(decision -> decision.reason() != null && !decision.reason().isBlank()));
    }

    @Test
    void cpuOnlyGraphUsesCpuPartitionTargetInAutoMode() {
        Tensor a = new Tensor(new double[]{1, 2, 3, 4}, new int[]{4}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{5, 6, 7, 8}, new int[]{4}, null, "b", DataType.FLOAT64);
        Tensor out = a.add(b).mul(a);

        graph.CompiledGraph compiled = graph.CompiledGraph.compile(out, CompileConfig.inference());

        assertEquals(
                List.of("CPU"),
                compiled.compileTrace().partitionPlanning().jobs().stream()
                        .map(trace.compile.PartitionCompileTrace.JobTrace::target)
                        .toList()
        );
        assertTrue(compiled.compileTrace().partitionPlanning().decisions().size() > 0);
    }

    @Test
    void autoPartitionTraceKeepsGpuAndCpuPlanningJobsWhenGpuNodesExist() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{3, 2}, null, "b", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor out = matmul.relu();

        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();

        backendIntentPlan = backendIntentPlan.withBackend(matmul, ComputeBackend.GPU_METAL);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_METAL);
        graph.CompiledGraph compiled = graph.CompiledGraph.compile(out, CompileConfig.inference(), backendIntentPlan);

        assertEquals(
                List.of("GPU_METAL", "CPU"),
                compiled.compileTrace().partitionPlanning().jobs().stream()
                        .map(trace.compile.PartitionCompileTrace.JobTrace::target)
                        .toList()
        );
    }

    @Test
    void acceleratorTraceAttributesIncludeBufferReasonAndStorageResidency() {
        Tensor a = new Tensor(new float[]{1f, 2f}, new int[]{2}, null, "traceA", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{3f, 4f}, new int[]{2}, null, "traceB", DataType.FLOAT32);
        Tensor out = a.add(b);
        graph.CompiledGraph compiled = graph.CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline());
        CompiledNode outputNode = publicationNode(compiled, out);
        SyntheticAcceleratorExecutable executable = new SyntheticAcceleratorExecutable(outputNode.id());
        CompiledNodeExecutionMetadata metadata = testsupport.MetadataArtifacts.acceleratorMetadata(ComputeBackend.GPU_CUDA, executable);
        PreparedExecutionStep step = new PreparedExecutionStep(outputNode, metadata);
        PreparedExecution prepared = new PreparedExecution(
                config.runtime.RuntimeConfig.inferenceDefaults(),
                false,
                List.of(step),
                List.of(step),
                List.of(),
                compiled.program().compiledNodes(),
                compiled.program().descriptorIndex(),
                compiled.publication(),
                outputNode,
                null,
                trace.prepare.PrepareTrace.skipped()
        );

        var trace = prepared.executeTraced(ExecutionMode.FORWARD);
        var attrs = trace.steps().getFirst().metadata().attributes();

        assertEquals("GPU_CUDA", attrs.get("acceleratorBufferBackend"));
        assertEquals("BUFFER_BINDING_AVAILABLE", attrs.get("acceleratorBufferReasonCode"));
        assertEquals("BUFFER_BINDING", attrs.get("acceleratorBufferExecutionPath"));
        assertEquals("CPU_ARRAY", attrs.get("storageResidency"));
    }

    @Test
    void gpuLoweredRegionRunTraceReferencesRegionIdOnly() {
        Tensor out = Tensor.scalar(1.0f).relu();
        graph.CompiledGraph compiled = graph.CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline());
        CompiledNode outputNode = publicationNode(compiled, out);
        GpuLoweredRegionManifest manifest = sampleManifest(outputNode.id());
        SyntheticAcceleratorExecutable executable = new SyntheticAcceleratorExecutable(outputNode.id(), manifest);
        CompiledNodeExecutionMetadata metadata = testsupport.MetadataArtifacts.acceleratorMetadata(ComputeBackend.GPU_CUDA, executable);
        PreparedExecutionStep step = new PreparedExecutionStep(outputNode, metadata);
        PreparedExecution prepared = new PreparedExecution(
                config.runtime.RuntimeConfig.inferenceDefaults(),
                false,
                List.of(step),
                List.of(step),
                List.of(),
                compiled.program().compiledNodes(),
                compiled.program().descriptorIndex(),
                compiled.publication(),
                outputNode,
                null,
                trace.prepare.PrepareTrace.skipped()
        );

        var attrs = prepared.executeTraced(ExecutionMode.FORWARD).steps().getFirst().metadata().attributes();

        assertEquals(manifest.regionId(), attrs.get("gpuLoweredRegionId"));
        assertFalse(attrs.containsKey("gpuLoweredRegionManifest"));
    }

    @Test
    void phaseNineteenPreparedTraceCarriesMultiOpRegionMetrics() {
        Tensor out = Tensor.scalar(1.0f).relu();
        graph.CompiledGraph compiled = graph.CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline());
        CompiledNode outputNode = publicationNode(compiled, out);
        GpuLoweredRegionManifest manifest = sampleMultiOpFusedManifest(outputNode.id());

        var attrs = syntheticTraceAttributes(
                out,
                outputNode,
                compiled,
                new SyntheticAcceleratorExecutable(outputNode.id(), manifest)
        );

        assertEquals(manifest.regionId(), attrs.get("gpuRegionId"));
        assertEquals(manifest.regionId(), attrs.get("gpuLoweredRegionId"));
        assertEquals(manifest.regionId(), attrs.get("regionId"));
        assertEquals("GPU_CUDA", attrs.get("regionTarget"));
        assertEquals("CUDA_GRAPH_REGION", attrs.get("loweringFamily"));
        assertEquals(outputNode.id(), attrs.get("anchorNodeId"));
        assertEquals(List.of(outputNode.id()), attrs.get("boundaryOutputNodeIds"));
        assertEquals(List.of("GRAPH_EXECUTABLE"), attrs.get("regionExecutionKindSummary"));
        assertEquals(List.of("DEVICE_BUFFER"), attrs.get("regionStorageContractSummary"));
        assertEquals(2, attrs.get("selectedRegionLength"));
        assertEquals(2, attrs.get("loweredPrimitiveCount"));
        assertEquals(1, attrs.get("gpuFusedSubpatternCount"));
        assertEquals(List.of("ELEMENTWISE_CHAIN"), attrs.get("gpuFusedSubpatternTypes"));
        assertEquals("BUFFER_BINDING", attrs.get("acceleratorBufferExecutionPath"));
        assertEquals("BUFFER_BINDING_AVAILABLE", attrs.get("acceleratorBufferReasonCode"));
        assertEquals(0, attrs.get("cpuMaterializationCount"));
        assertEquals(0, attrs.get("deviceHandoffCount"));
    }

    @Test
    void phaseNineteenTraceDistinguishesBufferBindingTensorArrayAndCpuFallback() {
        Tensor out = Tensor.scalar(1.0f).relu();
        graph.CompiledGraph compiled = graph.CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline());
        CompiledNode outputNode = publicationNode(compiled, out);

        var bufferAttrs = syntheticTraceAttributes(
                out,
                outputNode,
                compiled,
                new SyntheticAcceleratorExecutable(outputNode.id(), null)
        );
        var tensorArrayAttrs = syntheticTraceAttributes(
                out,
                outputNode,
                compiled,
                new SyntheticAcceleratorExecutable(
                        outputNode.id(),
                        null,
                        AcceleratorBufferExecutionPath.TENSOR_ARRAY,
                        AcceleratorBufferReasonCode.NATIVE_BUFFER_ABI_UNAVAILABLE
                )
        );
        var cpuFallbackAttrs = syntheticTraceAttributes(
                out,
                outputNode,
                compiled,
                new SyntheticAcceleratorExecutable(
                        outputNode.id(),
                        null,
                        AcceleratorBufferExecutionPath.CPU_FALLBACK,
                        AcceleratorBufferReasonCode.NATIVE_BUFFER_EXECUTION_FAILED
                )
        );

        assertEquals("BUFFER_BINDING", bufferAttrs.get("acceleratorBufferExecutionPath"));
        assertEquals("TENSOR_ARRAY", tensorArrayAttrs.get("acceleratorBufferExecutionPath"));
        assertEquals("CPU_FALLBACK", cpuFallbackAttrs.get("acceleratorBufferExecutionPath"));
        assertEquals("NATIVE_BUFFER_ABI_UNAVAILABLE", tensorArrayAttrs.get("acceleratorBufferReasonCode"));
        assertEquals("NATIVE_BUFFER_EXECUTION_FAILED", cpuFallbackAttrs.get("acceleratorBufferReasonCode"));
        assertFalse(bufferAttrs.containsKey("fallbackOccurred"));
        assertEquals(true, tensorArrayAttrs.get("fallbackOccurred"));
        assertEquals("ACCELERATOR_TENSOR_ARRAY_FALLBACK", tensorArrayAttrs.get("fallbackKind"));
        assertEquals(List.of("ACCELERATOR_TENSOR_ARRAY_FALLBACK"), tensorArrayAttrs.get("fallbackKinds"));
        assertEquals("NATIVE_BUFFER_ABI_UNAVAILABLE", tensorArrayAttrs.get("fallbackReasonCode"));
        assertEquals(List.of("NATIVE_BUFFER_ABI_UNAVAILABLE"), tensorArrayAttrs.get("fallbackReasonCodes"));
        assertEquals("synthetic accelerator buffer trace", tensorArrayAttrs.get("fallbackReason"));
        assertEquals(List.of("synthetic accelerator buffer trace"), tensorArrayAttrs.get("fallbackReasons"));
        assertEquals(true, cpuFallbackAttrs.get("fallbackOccurred"));
        assertEquals("ACCELERATOR_CPU_FALLBACK", cpuFallbackAttrs.get("fallbackKind"));
        assertEquals(List.of("ACCELERATOR_CPU_FALLBACK"), cpuFallbackAttrs.get("fallbackKinds"));
        assertEquals("NATIVE_BUFFER_EXECUTION_FAILED", cpuFallbackAttrs.get("fallbackReasonCode"));
        assertEquals(List.of("NATIVE_BUFFER_EXECUTION_FAILED"), cpuFallbackAttrs.get("fallbackReasonCodes"));
        assertEquals("synthetic accelerator buffer trace", cpuFallbackAttrs.get("fallbackReason"));
        assertEquals(List.of("synthetic accelerator buffer trace"), cpuFallbackAttrs.get("fallbackReasons"));
    }

    @Test
    void gpuCompoundElementwiseTraceContainsPatternAndDagNodeTypes() {
        Tensor a = new Tensor(new float[]{1f, -2f, 3f, -4f}, new int[]{4}, null, "traceChainA", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{0.5f, 1f, -1f, 2f}, new int[]{4}, null, "traceChainB", DataType.FLOAT32);
        Tensor add = a.add(b);
        Tensor relu = add.relu();
        Tensor out = relu.exp();
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(add, ComputeBackend.GPU_CUDA);
        backendIntentPlan = backendIntentPlan.withBackend(relu, ComputeBackend.GPU_CUDA);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_CUDA);
        graph.CompiledGraph compiled = graph.CompiledGraph.compile(out, CompileConfig.inference(), backendIntentPlan);
        PreparedExecution prepared = compiled.prepare(config.runtime.RuntimeConfig.inferenceDefaults());
        int addNodeId = nodeId(compiled, operations.Operation.OpType.ADD);
        int reluNodeId = nodeId(compiled, operations.Operation.OpType.RELU);
        var trace = prepared.executeTraced(ExecutionMode.FORWARD);
        var gpuStep = trace.steps().stream()
                .filter(step -> "GPU_CUDA".equals(step.backend()))
                .findFirst()
                .orElseThrow();
        var attrs = gpuStep.metadata().attributes();

        assertEquals("ELEMENTWISE_CHAIN", attrs.get("gpuCompoundPattern"));
        assertEquals(true, attrs.get("gpuCompoundSupported"));
        assertEquals("SUPPORTED", attrs.get("gpuCompoundReason"));
        assertTrue(((List<?>) attrs.get("gpuCompoundOrderedNodeIds")).containsAll(List.of(addNodeId, reluNodeId)));
        assertTrue(((List<?>) attrs.get("gpuCompoundDagNodeTypes")).containsAll(List.of("ADD", "RELU", "EXP")));
        if ("BUFFER_BINDING".equals(attrs.get("acceleratorBufferExecutionPath"))) {
            assertFalse(trace.cpuMaterializations().stream().anyMatch(materialization ->
                    (materialization.nodeId() == addNodeId || materialization.nodeId() == reluNodeId)
                            && materialization.reason() == CpuMaterializationReason.CPU_CONSUMER));
        }
    }

    @Test
    void metalDenseLossTrainingTraceKeepsForwardLossGpuOwnedWithoutInternalCpuConsumerMaterialization() {
        Tensor logits = new Tensor(new float[]{
                1f, 2f, 3f,
                1f, 0f, -1f
        }, new int[]{2, 3}, null, "traceDenseLossLogits", DataType.FLOAT32);
        logits.setRequiresGrad(true);
        Tensor targets = new Tensor(new float[]{
                0f, 0f, 1f,
                1f, 0f, 0f
        }, new int[]{2, 3}, null, "traceDenseLossTargets", DataType.FLOAT32);
        Tensor loss = logits.crossEntropyLoss(targets, 1);
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(loss, ComputeBackend.GPU_METAL);
        graph.CompiledGraph compiled = graph.CompiledGraph.compile(loss, CompileConfig.training(), backendIntentPlan);
        PreparedExecution prepared = compiled.prepare(config.runtime.RuntimeConfig.trainingDefaults());
        int lossNodeId = nodeId(compiled, operations.Operation.OpType.CROSS_ENTROPY_LOSS);
        var trace = prepared.executeTraced(ExecutionMode.FORWARD_BACKWARD);

        assertTrue(trace.steps().stream()
                .anyMatch(step -> step.backend().equals("GPU_METAL") && step.opType().equals("CROSS_ENTROPY_LOSS")));
        assertFalse(trace.cpuMaterializations().stream()
                .anyMatch(materialization -> materialization.nodeId() == lossNodeId
                        && materialization.reason() == CpuMaterializationReason.CPU_CONSUMER));
        assertFalse(trace.cpuMaterializations().stream()
                .anyMatch(materialization -> materialization.reason() == CpuMaterializationReason.CPU_FALLBACK));
    }

    @Test
    void traceRendersGpuFusedSubpatternSpanAndPrimitiveCount() {
        Tensor out = Tensor.scalar(1.0f).relu();
        graph.CompiledGraph compiled = graph.CompiledGraph.compile(out, CompileConfig.noGraphOptimizationBaseline());
        CompiledNode outputNode = publicationNode(compiled, out);
        GpuLoweredRegionManifest manifest = sampleFusedManifest(outputNode.id());
        SyntheticAcceleratorExecutable executable = new SyntheticAcceleratorExecutable(outputNode.id(), manifest);
        CompiledNodeExecutionMetadata metadata = testsupport.MetadataArtifacts.acceleratorMetadata(ComputeBackend.GPU_CUDA, executable);
        PreparedExecutionStep step = new PreparedExecutionStep(outputNode, metadata);
        PreparedExecution prepared = new PreparedExecution(
                config.runtime.RuntimeConfig.inferenceDefaults(),
                false,
                List.of(step),
                List.of(step),
                List.of(),
                compiled.program().compiledNodes(),
                compiled.program().descriptorIndex(),
                compiled.publication(),
                outputNode,
                null,
                trace.prepare.PrepareTrace.skipped()
        );

        var attrs = prepared.executeTraced(ExecutionMode.FORWARD).steps().getFirst().metadata().attributes();

        assertEquals(1, attrs.get("gpuFusedSubpatternCount"));
        assertEquals(List.of("ELEMENTWISE_CHAIN"), attrs.get("gpuFusedSubpatternTypes"));
        assertEquals(List.of(List.of(outputNode.id())), attrs.get("gpuFusedSubpatternOriginalNodeIds"));
        assertEquals(List.of(1), attrs.get("gpuFusedSubpatternLoweredPrimitiveCount"));
        assertEquals(List.of("SUPPORTED"), attrs.get("gpuFusedSubpatternReasons"));
        assertEquals("BUFFER_BINDING", attrs.get("acceleratorBufferDecision"));
        assertEquals("BUFFER_BINDING_AVAILABLE", attrs.get("acceleratorBufferReasonCode"));
    }

    @Test
    void prepareTraceRendersDTypeResidencyRejectionReasons() {
        GpuLoweredRegionManifest manifest = new GpuLoweredRegionManifest(
                "gpu-cuda-region-dtype",
                ComputeBackend.GPU_CUDA,
                70,
                List.of(70, 71),
                List.of(60),
                List.of(71),
                2,
                List.of(),
                List.of(),
                List.of(new GpuLoweredRegionValueAssumption(
                        60,
                        "input",
                        DataType.BOOL,
                        1,
                        List.of(4),
                        "CONTIGUOUS",
                        true,
                        false,
                        0L
                )),
                List.of(new GpuLoweredRegionValueAssumption(
                        71,
                        "output",
                        DataType.INT32,
                        1,
                        List.of(4),
                        "CONTIGUOUS",
                        true,
                        false,
                        0L
                )),
                GpuCompoundRegionSummary.none(ComputeBackend.GPU_CUDA, List.of(70, 71)),
                List.of(
                        new GpuLoweredRegionRejection(
                                "dtype_residency.compute",
                                70,
                                "p0",
                                "",
                                GpuLoweringUnsupportedReason.UNSUPPORTED_DTYPE,
                                "dtypeResidency backend=GPU_CUDA role=compute dtype=BFLOAT16 unsupported"
                        ),
                        new GpuLoweredRegionRejection(
                                "dtype_residency.output",
                                71,
                                "p1",
                                "",
                                GpuLoweringUnsupportedReason.UNSUPPORTED_DTYPE,
                                "dtypeResidency backend=GPU_METAL role=output dtype=INT32 unsupported"
                        )
                ),
                GpuLoweredRegionCandidateSpan.none(List.of(70, 71)),
                Map.of("dtypeResidency.input.60", "backend=GPU_METAL role=externalInput dtype=BOOL residentRepresentable=true")
        );
        var selection = new trace.prepare.BackendSelectionTrace(
                1,
                1,
                0,
                List.of(new trace.prepare.BackendSelectionDecisionTrace(
                        70,
                        List.of(70, 71),
                        List.of(ComputeBackend.GPU_CUDA.name()),
                        true,
                        ComputeBackend.GPU_CUDA.name(),
                        "selected",
                        128L,
                        null,
                        List.of(),
                        testsupport.TraceSnapshotTestSupport.traceManifest(manifest)
                ))
        );

        String rendered = GpuLoweredRegionTraceRenderer.renderCompact(
                selection.decisions().getFirst().gpuLoweredRegionManifest()
        );

        assertTrue(rendered.contains("dtypeResidency"));
        assertTrue(rendered.contains("UNSUPPORTED_DTYPE"));
        assertTrue(rendered.contains("backend=GPU_METAL"));
        assertTrue(rendered.contains("backend=GPU_CUDA"));
        assertTrue(rendered.contains("dtype=BFLOAT16"));
        assertTrue(rendered.contains("dtype=INT32"));
        assertTrue(rendered.contains("dtype=BOOL"));
    }

    @Test
    void prepareTraceRendersPhaseSeventeenNormAndLossEvidence() {
        GpuLoweredRegionManifest manifest = new GpuLoweredRegionManifest(
                "gpu-metal-region-phase17",
                ComputeBackend.GPU_METAL,
                80,
                List.of(80, 81),
                List.of(70),
                List.of(81),
                2,
                List.of(new GpuLoweredRegionOriginalOp(
                        81,
                        "LOG_SOFTMAX",
                        List.of(80),
                        List.of(81),
                        DataType.FLOAT32,
                        List.of(2, 3),
                        List.of("p0", "p1"),
                        List.of()
                )),
                List.of(
                        new GpuLoweredPrimitiveManifest(
                                "p0",
                                "SOFTMAX",
                                List.of(81),
                                List.of("external:0"),
                                "node:0",
                                DataType.FLOAT32,
                                List.of(2, 3),
                                List.of(GpuLoweringUnsupportedReason.SUPPORTED)
                        ),
                        new GpuLoweredPrimitiveManifest(
                                "p1",
                                "LOG",
                                List.of(81),
                                List.of("node:0"),
                                "node:1",
                                DataType.FLOAT32,
                                List.of(2, 3),
                                List.of(GpuLoweringUnsupportedReason.SUPPORTED)
                        )
                ),
                List.of(),
                List.of(),
                GpuCompoundRegionSummary.none(ComputeBackend.GPU_METAL, List.of(80, 81)),
                List.of(
                        new GpuLoweredRegionRejection(
                                "planner.normalization",
                                90,
                                "",
                                "",
                                GpuLoweringUnsupportedReason.UNSUPPORTED_LAYOUT,
                                "UNSUPPORTED_LAYOUT: GPU_METAL normalization inputs require dense layout family=NORMALIZATION target=layer_norm_small"
                        ),
                        new GpuLoweredRegionRejection(
                                "planner.loss",
                                91,
                                "",
                                "",
                                GpuLoweringUnsupportedReason.UNSUPPORTED_INDEX_SEMANTICS,
                                "UNSUPPORTED_INDEX_SEMANTICS: GPU_METAL index-target loss target out of range: 17 for classes=16 family=LOSS_ADJACENT target=transformer_block_hot_path"
                        )
                ),
                GpuLoweredRegionCandidateSpan.none(List.of(80, 81)),
                Map.of("phase17Target", "target=transformer_block_hot_path")
        );
        var selection = new trace.prepare.BackendSelectionTrace(
                1,
                1,
                0,
                List.of(new trace.prepare.BackendSelectionDecisionTrace(
                        81,
                        List.of(80, 81),
                        List.of(ComputeBackend.GPU_METAL.name()),
                        true,
                        ComputeBackend.GPU_METAL.name(),
                        "selected",
                        256L,
                        null,
                        List.of(),
                        testsupport.TraceSnapshotTestSupport.traceManifest(manifest)
                ))
        );

        String rendered = GpuLoweredRegionTraceRenderer.renderCompact(
                selection.decisions().getFirst().gpuLoweredRegionManifest()
        );

        assertTrue(rendered.contains("LOG_SOFTMAX"));
        assertTrue(rendered.contains("SOFTMAX"));
        assertTrue(rendered.contains("UNSUPPORTED_INDEX_SEMANTICS"));
        assertTrue(rendered.contains("family=LOSS_ADJACENT"));
        assertTrue(rendered.contains("UNSUPPORTED_LAYOUT"));
        assertTrue(rendered.contains("family=NORMALIZATION"));
        assertTrue(rendered.contains("target=layer_norm_small"));
        assertTrue(rendered.contains("target=transformer_block_hot_path"));
    }

    private static Map<String, Object> syntheticTraceAttributes(
            Tensor out,
            CompiledNode outputNode,
            graph.CompiledGraph compiled,
            SyntheticAcceleratorExecutable executable
    ) {
        CompiledNodeExecutionMetadata metadata = testsupport.MetadataArtifacts.acceleratorMetadata(executable.backend(), executable);
        PreparedExecutionStep step = new PreparedExecutionStep(outputNode, metadata);
        PreparedExecution prepared = new PreparedExecution(
                config.runtime.RuntimeConfig.inferenceDefaults(),
                false,
                List.of(step),
                List.of(step),
                List.of(),
                compiled.program().compiledNodes(),
                compiled.program().descriptorIndex(),
                compiled.publication(),
                outputNode,
                null,
                trace.prepare.PrepareTrace.skipped()
        );
        return prepared.executeTraced(ExecutionMode.FORWARD).steps().getFirst().metadata().attributes();
    }

    private record SyntheticAcceleratorExecutable(
            int nodeId,
            GpuLoweredRegionManifest gpuLoweredRegionManifest,
            RegionExecutionPlan regionExecutionPlan,
            AcceleratorBufferExecutionPath executionPath,
            AcceleratorBufferReasonCode reasonCode
    ) implements PreparedAcceleratorExecutable {
        private SyntheticAcceleratorExecutable(int nodeId) {
            this(nodeId, null);
        }

        private SyntheticAcceleratorExecutable(int nodeId, GpuLoweredRegionManifest gpuLoweredRegionManifest) {
            this(
                    nodeId,
                    gpuLoweredRegionManifest,
                    AcceleratorBufferExecutionPath.BUFFER_BINDING,
                    AcceleratorBufferReasonCode.BUFFER_BINDING_AVAILABLE
            );
        }

        private SyntheticAcceleratorExecutable(
                int nodeId,
                GpuLoweredRegionManifest gpuLoweredRegionManifest,
                AcceleratorBufferExecutionPath executionPath,
                AcceleratorBufferReasonCode reasonCode
        ) {
            this(
                    nodeId,
                    gpuLoweredRegionManifest,
                    gpuLoweredRegionManifest == null ? null : sampleRegionPlan(gpuLoweredRegionManifest),
                    executionPath,
                    reasonCode
            );
        }

        @Override
        public ComputeBackend backend() {
            return ComputeBackend.GPU_CUDA;
        }

        @Override
        public void execute(ExecutionContext context) {
            context.markCpuCurrent(nodeId, "synthetic accelerator trace output");
        }

        @Override
        public AcceleratorBufferDecision lastAcceleratorBufferDecision() {
            return new AcceleratorBufferDecision(
                    ComputeBackend.GPU_CUDA,
                    AcceleratorBufferBindingMode.AUTO,
                    executionPath,
                    executionPath == AcceleratorBufferExecutionPath.BUFFER_BINDING,
                    false,
                    reasonCode,
                    "synthetic accelerator buffer trace",
                    List.of(),
                    List.of()
            );
        }
    }

    private static RegionExecutionPlan sampleRegionPlan(GpuLoweredRegionManifest manifest) {
        return new RegionExecutionPlan(
                manifest.regionId(),
                PartitionTarget.fromBackend(manifest.backend()),
                manifest.backend() == ComputeBackend.GPU_METAL
                        ? LoweringFamily.METAL_GRAPH_REGION
                        : LoweringFamily.CUDA_GRAPH_REGION,
                manifest.anchorNodeId(),
                manifest.orderedNodeIds(),
                manifest.externalInputNodeIds(),
                manifest.outputNodeIds(),
                List.of(),
                List.of(new RegionExecutionGroup(
                        manifest.regionId() + "-group-0",
                        manifest.orderedNodeIds(),
                        RegionExecutionKind.GRAPH_EXECUTABLE,
                        manifest.backend().name() + "_GRAPH",
                        manifest.externalInputNodeIds(),
                        manifest.outputNodeIds(),
                        List.of(),
                        RegionStorageContract.DEVICE_BUFFER,
                        "synthetic-region-plan"
                )),
                RegionCost.ofWork(manifest.selectedRegionLength()),
                RegionDecision.selected("synthetic", "synthetic-region-plan"),
                EmptyRegionPayload.INSTANCE
        );
    }

    private static GpuLoweredRegionManifest sampleMultiOpFusedManifest(int nodeId) {
        return new GpuLoweredRegionManifest(
                "gpu-cuda-region-multi-op-" + nodeId,
                ComputeBackend.GPU_CUDA,
                nodeId,
                List.of(nodeId - 1, nodeId),
                List.of(),
                List.of(nodeId),
                2,
                List.of(
                        new GpuLoweredRegionOriginalOp(
                                nodeId - 1,
                                "ADD",
                                List.of(),
                                List.of(),
                                DataType.FLOAT32,
                                List.of(1),
                                List.of("p0"),
                                List.of()
                        ),
                        new GpuLoweredRegionOriginalOp(
                                nodeId,
                                "RELU",
                                List.of(nodeId - 1),
                                List.of(nodeId),
                                DataType.FLOAT32,
                                List.of(1),
                                List.of("p1"),
                                List.of()
                        )
                ),
                List.of(
                        new GpuLoweredPrimitiveManifest(
                                "p0",
                                "ADD",
                                List.of(nodeId - 1),
                                List.of("external:0"),
                                "node:0",
                                DataType.FLOAT32,
                                List.of(1),
                                List.of()
                        ),
                        new GpuLoweredPrimitiveManifest(
                                "p1",
                                "RELU",
                                List.of(nodeId),
                                List.of("node:0"),
                                "node:1",
                                DataType.FLOAT32,
                                List.of(1),
                                List.of()
                        )
                ),
                List.of(),
                List.of(),
                GpuCompoundRegionSummary.supported(
                        ComputeBackend.GPU_CUDA,
                        GpuCompoundPatternType.ELEMENTWISE_CHAIN,
                        List.of(nodeId - 1, nodeId),
                        List.of(),
                        List.of(nodeId),
                        List.of("ADD", "RELU"),
                        List.of(),
                        "synthetic Phase 19 multi-op fused subpattern"
                ),
                List.of(),
                GpuLoweredRegionCandidateSpan.none(List.of(nodeId - 1, nodeId)),
                Map.of("dagNodeCount", "2")
        );
    }

    private static GpuLoweredRegionManifest sampleManifest(int nodeId) {
        return new GpuLoweredRegionManifest(
                "gpu-cuda-region-" + nodeId,
                ComputeBackend.GPU_CUDA,
                nodeId,
                List.of(nodeId),
                List.of(),
                List.of(nodeId),
                1,
                List.of(new GpuLoweredRegionOriginalOp(
                        nodeId,
                        "RELU",
                        List.of(),
                        List.of(nodeId),
                        DataType.FLOAT32,
                        List.of(1),
                        List.of("p0"),
                        List.of()
                )),
                List.of(new GpuLoweredPrimitiveManifest(
                        "p0",
                        "RELU",
                        List.of(nodeId),
                        List.of("external:0"),
                        "node:0",
                        DataType.FLOAT32,
                        List.of(1),
                        List.of()
                )),
                List.of(new GpuLoweredRegionValueAssumption(
                        nodeId,
                        "input",
                        DataType.FLOAT32,
                        1,
                        List.of(1),
                        "CONTIGUOUS",
                        true,
                        false,
                        0L
                )),
                List.of(),
                GpuCompoundRegionSummary.none(ComputeBackend.GPU_CUDA, List.of(nodeId)),
                List.of(new GpuLoweredRegionRejection(
                        "primitive",
                        nodeId,
                        "p0",
                        "",
                        GpuLoweringUnsupportedReason.DAG_PRIMITIVE_UNSUPPORTED,
                        "synthetic rejection"
                )),
                GpuLoweredRegionCandidateSpan.none(List.of(nodeId)),
                Map.of("dagNodeCount", "1")
        );
    }

    private static GpuLoweredRegionManifest sampleFusedManifest(int nodeId) {
        return new GpuLoweredRegionManifest(
                "gpu-cuda-region-fused-" + nodeId,
                ComputeBackend.GPU_CUDA,
                nodeId,
                List.of(nodeId),
                List.of(),
                List.of(nodeId),
                1,
                List.of(new GpuLoweredRegionOriginalOp(
                        nodeId,
                        "RELU",
                        List.of(),
                        List.of(nodeId),
                        DataType.FLOAT32,
                        List.of(1),
                        List.of("p0"),
                        List.of()
                )),
                List.of(new GpuLoweredPrimitiveManifest(
                        "p0",
                        "RELU",
                        List.of(nodeId),
                        List.of("external:0"),
                        "node:0",
                        DataType.FLOAT32,
                        List.of(1),
                        List.of()
                )),
                List.of(),
                List.of(),
                GpuCompoundRegionSummary.supported(
                        ComputeBackend.GPU_CUDA,
                        GpuCompoundPatternType.ELEMENTWISE_CHAIN,
                        List.of(nodeId),
                        List.of(),
                        List.of(nodeId),
                        List.of("RELU"),
                        List.of(),
                        "synthetic fused subpattern"
                ),
                List.of(),
                GpuLoweredRegionCandidateSpan.none(List.of(nodeId)),
                Map.of("dagNodeCount", "1")
        );
    }

    private static int nodeId(graph.CompiledGraph compiled, operations.Operation.OpType opType) {
        return compiled.program().compiledNodes().stream()
                .filter(node -> node.operation() != null && node.operation().opType() == opType)
                .map(CompiledNode::id)
                .findFirst()
                .orElseThrow();
    }

    private static Tensor specialLogSoftmax(Tensor input, int dimension) {
        return TensorPrimitiveBuilder.unary(
                input,
                input.getShapeUnsafe().clone(),
                new operations.reduction.logSoftmax(dimension),
                "legacyLogSoftmax",
                input.getDataType()
        );
    }

    private static CompiledNode compiledNode(graph.CompiledGraph compiled, int nodeId) {
        return compiled.program().compiledNodes().stream()
                .filter(node -> node.id() == nodeId)
                .findFirst()
                .orElseThrow();
    }

    private static CompiledNode publicationNode(graph.CompiledGraph compiled, Tensor tensor) {
        Integer nodeId = compiled.publication().nodeIdsByPublicationTarget().get(tensor);
        if (nodeId == null) {
            throw new IllegalStateException("Missing publication node for tensor " + tensor.getLabel());
        }
        return compiledNode(compiled, nodeId);
    }

    private static PartitionPlanningContext planningContext(graph.CompiledGraph compiled) {
        List<CompiledNode> nodes = compiled.program().compiledNodes();
        java.util.Map<Integer, java.util.List<CompiledNode>> consumers = new java.util.HashMap<>();
        for (CompiledNode node : nodes) {
            consumers.computeIfAbsent(node.id(), ignored -> new java.util.ArrayList<>());
        }
        for (CompiledNode node : nodes) {
            for (int inputId : node.inputIds()) {
                consumers.computeIfAbsent(inputId, ignored -> new java.util.ArrayList<>()).add(node);
            }
        }
        return new PartitionPlanningContext(
                false,
                nodes,
                CompiledTensorDescriptorBuilder.build(nodes),
                consumers
        );
    }
}
