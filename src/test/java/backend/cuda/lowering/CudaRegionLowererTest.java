package backend.cuda.lowering;

import planning.descriptor.CompiledTensorDescriptorBuilder;
import planning.descriptor.CompiledTensorDescriptorIndex;
import planning.intent.BackendIntentPlan;

import backend.contract.ComputeBackend;
import backend.accelerator.lowering.GpuCompoundLoweringArtifact;
import backend.accelerator.lowering.GpuCompoundPatternType;
import backend.accelerator.lowering.GpuLoweringCoverageMatrix;
import backend.lowering.BackendCapabilities;
import backend.lowering.LoweringContext;
import backend.lowering.LoweringRequest;
import backend.lowering.LoweringResult;
import backend.lowering.region.CudaRegionPayload;
import backend.lowering.region.RegionExecutionKind;
import backend.lowering.region.RegionExecutionPlan;
import config.optimizer.FuseConfig;
import config.runtime.RuntimeConfig;
import graph.compile.CompiledNodeSnapshotter;
import graph.model.CompiledNode;
import trace.compile.PartitionDecisionTrace;
import planning.memory.MemoryPlanner;
import planning.partition.Partition;
import planning.partition.PartitionBoundaryReason;
import planning.partition.PartitionEdge;
import planning.partition.PartitionPlannerStrategy;
import planning.partition.PartitionTarget;
import planning.partition.PartitionValue;
import planning.value.GraphValueRef;
import planning.partition.PartitionPlanningContext;
import backend.accelerator.dag.AcceleratorDagInput;
import backend.accelerator.dag.AcceleratorDagNode;
import backend.accelerator.dag.AcceleratorDagNodeType;
import backend.accelerator.dag.AcceleratorDagSpec;
import backend.accelerator.dag.AcceleratorDagValueRef;
import backend.accelerator.dag.AcceleratorSubgraphOp;
import backend.accelerator.dag.AcceleratorSubgraphSpec;
import planning.region.DefaultRegionPlanner;
import planning.region.ExecutionUnitKind;
import planning.region.PlannedRegion;
import planning.region.RegionPlanningContext;
import operations.Operation;
import operations.index.gatherGrad;
import operations.index.takeAlongAxisGrad;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.internal.TensorPrimitiveBuilder;
import tensor.options.AttentionOptions;
import tensor.options.Conv2dOptions;
import tensor.options.Pool2dOptions;

import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class CudaRegionLowererTest {
    @Test
    void cudaLowersLinearBiasReluAsLinearBiasActivationCompoundRegion() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "cudaLinearInput", DataType.FLOAT32);
        Tensor weight = new Tensor(new float[]{
                1f, 0f, 0f, 1f,
                0f, 1f, 1f, 0f,
                1f, 1f, 0f, 0f
        }, new int[]{3, 4}, null, "cudaLinearWeight", DataType.FLOAT32);
        Tensor bias = new Tensor(new float[]{0.5f, -0.5f, 1f, -1f}, new int[]{4}, null, "cudaLinearBias", DataType.FLOAT32);
        Tensor linear = input.linear(weight, bias);
        Tensor out = linear.relu();
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(linear, ComputeBackend.GPU_CUDA);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_CUDA);
        List<Tensor> graph = out.topologicalSort();
        List<CompiledNode> compiledNodes = CompiledNodeSnapshotter.snapshot(graph, backendIntentPlan);
        PartitionPlanningContext context = new PartitionPlanningContext(
                false,
                compiledNodes,
                CompiledTensorDescriptorBuilder.build(compiledNodes),
                consumers(compiledNodes)
        );
        int linearNodeId = nodeId(context, operations.Operation.OpType.LINEAR);
        int reluNodeId = nodeId(context, operations.Operation.OpType.RELU);
        CudaGpuBackendPartitionCapability adapter = new CudaGpuBackendPartitionCapability();
        var candidate = adapter.createCandidate(
                Set.of(linearNodeId, reluNodeId),
                context,
                Set.of(GraphValueRef.node(reluNodeId))
        );
        assertNotNull(candidate);
        CudaGpuPartitionPlan plan = (CudaGpuPartitionPlan) adapter.createPlan(candidate, context);

        assertNotNull(plan);
        assertEquals(GpuCompoundPatternType.LINEAR_BIAS_ACTIVATION, plan.compoundSummary().patternType());
        assertTrue(plan.compoundSummary().supported());
        assertTrue(plan.nodeIds().containsAll(List.of(linearNodeId, reluNodeId)));
        assertEquals(ComputeBackend.GPU_CUDA, plan.manifest().backend());
        assertTrue(!plan.manifest().loweredPrimitives().isEmpty());
        assertTrue(plan.dagSpec().nodes().stream().anyMatch(node -> node.type() == AcceleratorDagNodeType.LINEAR));
        assertTrue(plan.dagSpec().nodes().stream().anyMatch(node -> node.type() == AcceleratorDagNodeType.RELU));

        Partition partition = partition(
                "cuda-linear-bias-activation",
                PartitionTarget.GPU_CUDA,
                candidate.orderedNodeIds(),
                candidate.externalInputIds(),
                candidate.outputNodeIds().stream().map(GraphValueRef::node).toList(),
                List.of(GraphValueRef.node(reluNodeId))
        );
        PlannedRegion region = new DefaultRegionPlanner().planRegion(
                partition,
                new RegionPlanningContext(compiledNodes, FuseConfig.inferenceDefaults())
        );
        LoweringResult result = new CudaRegionLowerer().lower(new LoweringRequest(
                region,
                MemoryPlanner.plan(graph),
                new BackendCapabilities(Set.of(ComputeBackend.GPU_CUDA)),
                new LoweringContext(RuntimeConfig.inferenceDefaults(), compiledNodes, CompiledTensorDescriptorBuilder.build(compiledNodes), java.util.Map.of(partition.partitionId(), plan))
        ));

        assertNotNull(result);
        RegionExecutionPlan regionPlan = result.loweredRegion().units().getFirst().requireRegionPlan();
        assertEquals(backend.lowering.LoweringFamily.CUDA_GRAPH_REGION, regionPlan.loweringFamily());
        assertTrue(regionPlan.backendPayload() instanceof CudaRegionPayload);
        assertEquals(List.of(RegionExecutionKind.GRAPH_EXECUTABLE), regionPlan.executionGroups().stream()
                .map(group -> group.executionKind())
                .distinct()
                .toList());
        GpuCompoundLoweringArtifact artifact = result.loweredRegion().units().getFirst().requireArtifact(GpuCompoundLoweringArtifact.class);
        assertEquals(GpuCompoundPatternType.LINEAR_BIAS_ACTIVATION, artifact.summary().patternType());
        assertTrue(artifact.summary().orderedNodeIds().containsAll(List.of(linearNodeId, reluNodeId)));
        assertTrue(artifact.units().stream().anyMatch(unit ->
                unit.kind() == ExecutionUnitKind.UNIT_KERNEL
                        && unit.orderedNodeIds().contains(linearNodeId)));
        assertTrue(artifact.units().stream()
                .flatMap(unit -> unit.traceEvents().stream())
                .anyMatch(event -> event.contains("region-unit-node:")));
    }

    @Test
    void phaseNineteenCudaLowererKeepsMultiOpRegionAsSingleGraphUnit() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "phase19CudaA", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 0f, 0f, 1f, 1f, 1f}, new int[]{3, 2}, null, "phase19CudaB", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor relu = matmul.relu();
        Tensor out = relu.exp();
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(matmul, ComputeBackend.GPU_CUDA);
        backendIntentPlan = backendIntentPlan.withBackend(relu, ComputeBackend.GPU_CUDA);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_CUDA);
        List<Tensor> graph = out.topologicalSort();
        List<CompiledNode> compiledNodes = CompiledNodeSnapshotter.snapshot(graph, backendIntentPlan);
        PartitionPlanningContext planningContext = new PartitionPlanningContext(
                false,
                compiledNodes,
                CompiledTensorDescriptorBuilder.build(compiledNodes),
                consumers(compiledNodes)
        );
        int matmulNodeId = nodeId(planningContext, Operation.OpType.MATMUL);
        int reluNodeId = nodeId(planningContext, Operation.OpType.RELU);
        int expNodeId = nodeId(planningContext, Operation.OpType.EXP);
        List<Integer> selectedNodeIds = List.of(matmulNodeId, reluNodeId, expNodeId);
        CudaGpuBackendPartitionCapability adapter = new CudaGpuBackendPartitionCapability();
        var candidate = adapter.createCandidate(
                Set.copyOf(selectedNodeIds),
                planningContext,
                Set.of(GraphValueRef.node(expNodeId))
        );
        assertNotNull(candidate);
        CudaGpuPartitionPlan attachedPlan = (CudaGpuPartitionPlan) adapter.createPlan(candidate, planningContext);
        assertNotNull(attachedPlan);
        Partition partition = partition(
                "phase19-cuda-multi-op",
                PartitionTarget.GPU_CUDA,
                candidate.orderedNodeIds(),
                candidate.externalInputIds(),
                candidate.outputNodeIds().stream().map(GraphValueRef::node).toList(),
                List.of(GraphValueRef.node(expNodeId))
        );
        PlannedRegion region = new DefaultRegionPlanner().planRegion(
                partition,
                new RegionPlanningContext(compiledNodes, FuseConfig.inferenceDefaults())
        );

        LoweringResult result = new CudaRegionLowerer().lower(new LoweringRequest(
                region,
                MemoryPlanner.plan(graph),
                new BackendCapabilities(Set.of(ComputeBackend.GPU_CUDA)),
                new LoweringContext(RuntimeConfig.inferenceDefaults(), compiledNodes, CompiledTensorDescriptorBuilder.build(compiledNodes), java.util.Map.of(partition.partitionId(), attachedPlan))
        ));

        assertNotNull(result);
        assertEquals(1, result.loweredRegion().units().size());
        assertEquals(backend.lowering.LoweringFamily.CUDA_GRAPH_REGION, result.loweredRegion().units().getFirst().loweringFamily());
        assertTrue(attachedPlan.manifest().selectedRegionLength() > 1);
        assertTrue(attachedPlan.manifest().loweredPrimitives().size() > 1);
        assertTrue(result.loweredRegion().units().getFirst().orderedNodeIds().containsAll(selectedNodeIds));
        GpuCompoundLoweringArtifact artifact = result.loweredRegion().units().getFirst().requireArtifact(GpuCompoundLoweringArtifact.class);
        assertEquals(GpuCompoundPatternType.NONE, artifact.summary().patternType());
        assertTrue(artifact.units().stream().anyMatch(unit -> unit.kind() == ExecutionUnitKind.UNIT_KERNEL));
        assertTrue(artifact.units().stream().anyMatch(unit -> unit.kind() == ExecutionUnitKind.FUSED_ELEMENTWISE));
        assertTrue(artifact.units().stream()
                .flatMap(unit -> unit.traceEvents().stream())
                .anyMatch(event -> event.contains("region-unit-node:")));
    }

    @Test
    void cudaPlannerSupportMatchesSharedCoverageMatrixForForwardOps() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "cudaMatrixA", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{3, 2}, null, "cudaMatrixB", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor relu = matmul.relu();
        Tensor out = relu.exp();
        Tensor logSoftmax = specialLogSoftmax(out, 1);
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(matmul, ComputeBackend.GPU_CUDA);
        backendIntentPlan = backendIntentPlan.withBackend(relu, ComputeBackend.GPU_CUDA);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_CUDA);
        backendIntentPlan = backendIntentPlan.withBackend(logSoftmax, ComputeBackend.GPU_CUDA);
        PartitionPlanningContext context = planningContext(logSoftmax, backendIntentPlan);
        for (operations.Operation.OpType opType : List.of(operations.Operation.OpType.MATMUL, operations.Operation.OpType.RELU, operations.Operation.OpType.EXP, operations.Operation.OpType.LOG_SOFTMAX)) {
            assertTrue(GpuLoweringCoverageMatrix.isSupported(ComputeBackend.GPU_CUDA, opType));
            assertEquals("", CudaGpuBackendPartitionCapability.plannerUnsupportedReason(context.compiledNode(nodeId(context, opType)), context));
        }
    }

    @Test
    void cudaReductionIsPlannerSupported() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "cudaReductionInput", DataType.FLOAT32);
        Tensor out = input.sum(1);
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_CUDA);
        PartitionPlanningContext context = planningContext(out, backendIntentPlan);
        String reason = CudaGpuBackendPartitionCapability.plannerUnsupportedReason(context.compiledNode(nodeId(context, operations.Operation.OpType.SUM)), context);

        assertEquals("", reason);
    }

    @Test
    void cudaWindowLayoutPrimitivesLowerToDedicatedDagNodes() {
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();

        Tensor axisInput = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{4}, null, "cudaAxisInput", DataType.FLOAT32);
        Tensor axisOut = axisInput.unfold(0, 2, 2);
        backendIntentPlan = backendIntentPlan.withBackend(axisOut, ComputeBackend.GPU_CUDA);
        PartitionPlanningContext axisContext = planningContext(axisOut, backendIntentPlan);
        CompiledNode axisNode = axisContext.compiledNode(nodeId(axisContext, Operation.OpType.UNFOLD_AXIS));
        assertEquals("", CudaGpuBackendPartitionCapability.plannerUnsupportedReason(axisNode, axisContext));

        CudaGpuBackendPartitionCapability adapter = new CudaGpuBackendPartitionCapability();
        CudaGpuPartitionPlan axisPlan = (CudaGpuPartitionPlan) adapter.createPlan(
                adapter.createCandidate(Set.of(axisNode.id()), axisContext, Set.of(GraphValueRef.node(axisNode.id()))),
                axisContext
        );
        assertNotNull(axisPlan);
        assertEquals(AcceleratorDagNodeType.UNFOLD_AXIS, axisPlan.dagSpec().nodes().getFirst().type());

        Tensor image = new Tensor(new float[]{
                1f, 2f, 3f,
                4f, 5f, 6f,
                7f, 8f, 9f
        }, new int[]{1, 1, 3, 3}, null, "cudaWindowImage", DataType.FLOAT32);
        tensor.options.Window2dOptions window = tensor.options.Window2dOptions.of(2, 2);
        Tensor columns = image.unfold2d(window);
        backendIntentPlan = backendIntentPlan.withBackend(columns, ComputeBackend.GPU_CUDA);
        PartitionPlanningContext unfoldContext = planningContext(columns, backendIntentPlan);
        CompiledNode unfoldNode = unfoldContext.compiledNode(nodeId(unfoldContext, Operation.OpType.UNFOLD2D));
        assertEquals("", CudaGpuBackendPartitionCapability.plannerUnsupportedReason(unfoldNode, unfoldContext));
        CudaGpuPartitionPlan unfoldPlan = (CudaGpuPartitionPlan) adapter.createPlan(
                adapter.createCandidate(Set.of(unfoldNode.id()), unfoldContext, Set.of(GraphValueRef.node(unfoldNode.id()))),
                unfoldContext
        );
        assertNotNull(unfoldPlan);
        assertEquals(AcceleratorDagNodeType.UNFOLD2D, unfoldPlan.dagSpec().nodes().getFirst().type());

        Tensor folded = columns.fold2d(new int[]{1, 1, 3, 3}, window);
        backendIntentPlan = backendIntentPlan.withBackend(folded, ComputeBackend.GPU_CUDA);
        PartitionPlanningContext foldContext = planningContext(folded, backendIntentPlan);
        CompiledNode foldNode = foldContext.compiledNode(nodeId(foldContext, Operation.OpType.FOLD2D));
        assertEquals("", CudaGpuBackendPartitionCapability.plannerUnsupportedReason(foldNode, foldContext));
        CudaGpuPartitionPlan foldPlan = (CudaGpuPartitionPlan) adapter.createPlan(
                adapter.createCandidate(Set.of(foldNode.id()), foldContext, Set.of(GraphValueRef.node(foldNode.id()))),
                foldContext
        );
        assertNotNull(foldPlan);
        assertEquals(AcceleratorDagNodeType.FOLD2D, foldPlan.dagSpec().nodes().getFirst().type());
    }

    @Test
    void cudaSupportedNormalizationUsesSharedCoverageReason() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "cudaNormInput", DataType.FLOAT32);
        Tensor gamma = new Tensor(new float[]{1f, 1f}, new int[]{2}, null, "cudaNormGamma", DataType.FLOAT32);
        Tensor beta = new Tensor(new float[]{0f, 0f}, new int[]{2}, null, "cudaNormBeta", DataType.FLOAT32);
        Tensor out = input.layerNorm(gamma, beta, 1.0e-5);
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_CUDA);
        PartitionPlanningContext context = planningContext(out, backendIntentPlan);
        String reason = CudaGpuBackendPartitionCapability.plannerUnsupportedReason(context.compiledNode(nodeId(context, operations.Operation.OpType.LAYER_NORM)), context);

        assertEquals("", reason);
    }

    @Test
    void cudaDirectForwardSdpaRemainsCapabilityMissingUntilNativeEvidenceExists() {
        Tensor q = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "cudaSdpaQ", DataType.FLOAT32);
        Tensor k = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "cudaSdpaK", DataType.FLOAT32);
        Tensor v = new Tensor(new float[]{10f, 1f, 1f, 10f}, new int[]{1, 2, 2}, null, "cudaSdpaV", DataType.FLOAT32);
        Tensor out = specialSdpa(q, k, v, null, 0.5d);
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_CUDA);
        PartitionPlanningContext context = planningContext(out, backendIntentPlan);
        String reason = CudaGpuBackendPartitionCapability.plannerUnsupportedReason(
                context.compiledNode(nodeId(context, Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION)),
                context
        );

        assertTrue(reason.contains("CAPABILITY_MISSING"));
        assertTrue(reason.contains("CUDA direct forward SDPA"));
        assertTrue(reason.contains("target=transformer_block_hot_path"));
        assertTrue(reason.contains("maskMode=UNMASKED"));
        assertFalse(GpuLoweringCoverageMatrix.isSupported(ComputeBackend.GPU_CUDA, Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION));
    }

    @Test
    void cudaMaskedForwardSdpaUsesStableMaskRejectionReason() {
        Tensor q = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "cudaMaskedSdpaQ", DataType.FLOAT32);
        Tensor k = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "cudaMaskedSdpaK", DataType.FLOAT32);
        Tensor v = new Tensor(new float[]{10f, 1f, 1f, 10f}, new int[]{1, 2, 2}, null, "cudaMaskedSdpaV", DataType.FLOAT32);
        Tensor mask = new Tensor(new byte[]{1, 0, 1, 1}, new int[]{1, 2, 2}, null, "cudaMaskedSdpaMask", DataType.BOOL);
        Tensor out = specialSdpa(q, k, v, mask, 0.5d);
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_CUDA);
        PartitionPlanningContext context = planningContext(out, backendIntentPlan);
        String reason = CudaGpuBackendPartitionCapability.plannerUnsupportedReason(
                context.compiledNode(nodeId(context, Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION)),
                context
        );

        assertContainsAll(
                reason,
                "CAPABILITY_MISSING",
                "CUDA direct forward SDPA",
                "target=masked_sdpa_small",
                "maskMode=EXTERNAL_BOOL_MASK"
        );
    }

    @Test
    void cudaCausalForwardSdpaReportsMaskModeBeforeCapabilityMissing() {
        Tensor q = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "cudaCausalSdpaQ", DataType.FLOAT32);
        Tensor k = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "cudaCausalSdpaK", DataType.FLOAT32);
        Tensor v = new Tensor(new float[]{10f, 1f, 1f, 10f}, new int[]{1, 2, 2}, null, "cudaCausalSdpaV", DataType.FLOAT32);
        Tensor out = specialSdpa(q, k, v, causalMask(), 0.5d);
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_CUDA);
        PartitionPlanningContext context = planningContext(out, backendIntentPlan);
        String reason = CudaGpuBackendPartitionCapability.plannerUnsupportedReason(
                context.compiledNode(nodeId(context, Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION)),
                context
        );

        assertContainsAll(
                reason,
                "CAPABILITY_MISSING",
                "target=masked_sdpa_small",
                "maskMode=CAUSAL_BOOL_MASK"
        );
    }

    @Test
    void cudaExternalAndCausalForwardSdpaReportsCombinedMaskMode() {
        Tensor q = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "cudaExternalCausalSdpaQ", DataType.FLOAT32);
        Tensor k = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "cudaExternalCausalSdpaK", DataType.FLOAT32);
        Tensor v = new Tensor(new float[]{10f, 1f, 1f, 10f}, new int[]{1, 2, 2}, null, "cudaExternalCausalSdpaV", DataType.FLOAT32);
        Tensor mask = new Tensor(new byte[]{1, 0, 1, 1}, new int[]{1, 2, 2}, null, "cudaExternalCausalSdpaMask", DataType.BOOL);
        Tensor out = specialSdpa(q, k, v, mask.logicalAnd(causalMask()), 0.5d);
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_CUDA);
        PartitionPlanningContext context = planningContext(out, backendIntentPlan);
        String reason = CudaGpuBackendPartitionCapability.plannerUnsupportedReason(
                context.compiledNode(nodeId(context, Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION)),
                context
        );

        assertContainsAll(
                reason,
                "CAPABILITY_MISSING",
                "target=masked_sdpa_small",
                "maskMode=EXTERNAL_AND_CAUSAL_BOOL_MASK"
        );
    }

    @Test
    void cudaForwardSdpaReportsDtypeAndLayoutBeforeCapabilityMissing() {
        Tensor q64 = new Tensor(new double[]{1d, 0d, 0d, 1d}, new int[]{1, 2, 2}, null, "cudaSdpaQ64", DataType.FLOAT64);
        Tensor k64 = new Tensor(new double[]{1d, 0d, 0d, 1d}, new int[]{1, 2, 2}, null, "cudaSdpaK64", DataType.FLOAT64);
        Tensor v64 = new Tensor(new double[]{10d, 1d, 1d, 10d}, new int[]{1, 2, 2}, null, "cudaSdpaV64", DataType.FLOAT64);
        Tensor dtypeOut = specialSdpa(q64, k64, v64, null, 0.5d);
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(dtypeOut, ComputeBackend.GPU_CUDA);
        PartitionPlanningContext dtypeContext = planningContext(dtypeOut, backendIntentPlan);

        String dtypeReason = CudaGpuBackendPartitionCapability.plannerUnsupportedReason(
                dtypeContext.compiledNode(nodeId(dtypeContext, Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION)),
                dtypeContext
        );
        assertTrue(dtypeReason.contains("UNSUPPORTED_DTYPE"));

        Tensor baseQ = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "cudaSdpaBaseQ", DataType.FLOAT32);
        Tensor qView = baseQ.permute(0, 2, 1);
        Tensor k = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "cudaSdpaDenseK", DataType.FLOAT32);
        Tensor v = new Tensor(new float[]{10f, 1f, 1f, 10f}, new int[]{1, 2, 2}, null, "cudaSdpaDenseV", DataType.FLOAT32);
        Tensor layoutOut = specialSdpa(qView, k, v, null, 0.5d);
        backendIntentPlan = backendIntentPlan.withBackend(layoutOut, ComputeBackend.GPU_CUDA);
        PartitionPlanningContext layoutContext = planningContext(layoutOut, backendIntentPlan);

        String layoutReason = CudaGpuBackendPartitionCapability.plannerUnsupportedReason(
                layoutContext.compiledNode(nodeId(layoutContext, Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION)),
                layoutContext
        );
        assertTrue(layoutReason.contains("UNSUPPORTED_LAYOUT"));
    }

    @Test
    void cudaGpuFusedOpTypeRejectsWithStableCompoundReason() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{4}, null, "cudaFusedInput", DataType.FLOAT32);
        Tensor out = TensorPrimitiveBuilder.unary(input, new SyntheticFusedOperation(), "cudaCpuFusedOp", DataType.FLOAT32);
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_CUDA);
        PartitionPlanningContext context = planningContext(out, backendIntentPlan);
        String reason = CudaGpuBackendPartitionCapability.plannerUnsupportedReason(context.compiledNode(nodeId(context, Operation.OpType.FUSED)), context);

        assertTrue(reason.contains("CPU_FUSED_OPERATION_UNSUPPORTED"));
        assertTrue(reason.contains("operation FUSED is not supported by GPU_CUDA lowering"));
    }

    @Test
    void cudaUnsupportedLossAdjacentUsesSharedUnsupportedReason() {
        Tensor logits = new Tensor(new float[]{1f, 2f, 3f, 1f, 0f, -1f}, new int[]{2, 3}, null, "cudaLossLogits", DataType.FLOAT32);
        Tensor targetIndices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "cudaLossTargets", DataType.INT32);
        Tensor out = logits.crossEntropyLossFromIndices(targetIndices, 1);
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_CUDA);
        PartitionPlanningContext context = planningContext(out, backendIntentPlan);
        String reason = CudaGpuBackendPartitionCapability.plannerUnsupportedReason(context.compiledNode(nodeId(context, operations.Operation.OpType.CROSS_ENTROPY_LOSS_INDICES)), context);

        assertTrue(reason.contains("UNSUPPORTED_INDEX_SEMANTICS"));
        assertTrue(reason.contains("operation CROSS_ENTROPY_LOSS_INDICES is not supported by GPU_CUDA lowering"));
    }

    @Test
    void cudaDenseLossValidatesContractBeforeDagPrimitiveMissing() {
        Tensor logits = new Tensor(new float[]{1f, 2f, 3f, 1f, 0f, -1f}, new int[]{2, 3}, null, "cudaDenseCeLogits", DataType.FLOAT32);
        Tensor denseTargets = new Tensor(new float[]{0f, 0f, 1f, 1f, 0f, 0f}, new int[]{2, 3}, null, "cudaDenseCeTargets", DataType.FLOAT32);
        Tensor crossEntropy = logits.crossEntropyLoss(denseTargets, 1);
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(crossEntropy, ComputeBackend.GPU_CUDA);
        PartitionPlanningContext ceContext = planningContext(crossEntropy, backendIntentPlan);

        String ceReason = CudaGpuBackendPartitionCapability.plannerUnsupportedReason(
                ceContext.compiledNode(nodeId(ceContext, Operation.OpType.CROSS_ENTROPY_LOSS)),
                ceContext
        );

        assertContainsAll(
                ceReason,
                "DAG_PRIMITIVE_UNSUPPORTED",
                "operation CROSS_ENTROPY_LOSS is not supported by GPU_CUDA lowering",
                "family=LOSS_ADJACENT",
                "target=dense_loss_small"
        );

        Tensor logProbs = new Tensor(new float[]{-2f, -1f, -0.5f, -0.25f}, new int[]{2, 2}, null, "cudaDenseNllLogProbs", DataType.FLOAT32);
        Tensor nllTargets = new Tensor(new float[]{0f, 1f, 1f, 0f}, new int[]{2, 2}, null, "cudaDenseNllTargets", DataType.FLOAT32);
        Tensor nll = logProbs.nllLoss(nllTargets, 1);
        backendIntentPlan = backendIntentPlan.withBackend(nll, ComputeBackend.GPU_CUDA);
        PartitionPlanningContext nllContext = planningContext(nll, backendIntentPlan);

        String nllReason = CudaGpuBackendPartitionCapability.plannerUnsupportedReason(
                nllContext.compiledNode(nodeId(nllContext, Operation.OpType.NLL_LOSS)),
                nllContext
        );

        assertContainsAll(
                nllReason,
                "DAG_PRIMITIVE_UNSUPPORTED",
                "operation NLL_LOSS is not supported by GPU_CUDA lowering",
                "family=LOSS_ADJACENT",
                "target=dense_loss_small"
        );
    }

    @Test
    void cudaPhaseSeventeenKeepsDirectNonDenseLayoutRejectionBeforeExecution() {
        Tensor base = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "cudaBase", DataType.FLOAT32);
        Tensor nonDense = base.permute(1, 0);
        Tensor rhs = new Tensor(new float[]{1f, 1f, 1f, 1f, 1f, 1f}, new int[]{3, 2}, null, "cudaRhs", DataType.FLOAT32);
        Tensor out = nonDense.add(rhs);
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_CUDA);
        PartitionPlanningContext context = planningContext(out, backendIntentPlan);
        String reason = CudaGpuBackendPartitionCapability.plannerUnsupportedReason(context.compiledNode(nodeId(context, operations.Operation.OpType.ADD)), context);

        assertEquals(
                "UNSUPPORTED_LAYOUT: direct non-dense CUDA compute remains conservative until metadata-only view propagation or dense materialization makes the consumer layout legal",
                reason
        );
    }

    @Test
    void cudaPhaseSeventeenReductionAndNormReasonsIncludeMatrixDetail() {
        Tensor reductionInput = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "cudaPhase17ReductionInput", DataType.FLOAT32);
        Tensor sum = reductionInput.sum(1);
        Tensor mean = reductionInput.mean(1);
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(sum, ComputeBackend.GPU_CUDA);
        backendIntentPlan = backendIntentPlan.withBackend(mean, ComputeBackend.GPU_CUDA);
        PartitionPlanningContext sumContext = planningContext(sum, backendIntentPlan);
        PartitionPlanningContext meanContext = planningContext(mean, backendIntentPlan);

        assertEquals("", CudaGpuBackendPartitionCapability.plannerUnsupportedReason(sumContext.compiledNode(nodeId(sumContext, Operation.OpType.SUM)), sumContext));
        assertEquals("", CudaGpuBackendPartitionCapability.plannerUnsupportedReason(meanContext.compiledNode(nodeId(meanContext, Operation.OpType.MEAN)), meanContext));

        Tensor normInput = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "cudaPhase17NormInput", DataType.FLOAT32);
        Tensor gamma = new Tensor(new float[]{1f, 1f}, new int[]{2}, null, "cudaPhase17NormGamma", DataType.FLOAT32);
        Tensor beta = new Tensor(new float[]{0f, 0f}, new int[]{2}, null, "cudaPhase17NormBeta", DataType.FLOAT32);
        Tensor layerNorm = normInput.layerNorm(gamma, beta, 1.0e-5);
        Tensor rmsNorm = normInput.rmsNorm(gamma, 1.0e-5);
        backendIntentPlan = backendIntentPlan.withBackend(layerNorm, ComputeBackend.GPU_CUDA);
        backendIntentPlan = backendIntentPlan.withBackend(rmsNorm, ComputeBackend.GPU_CUDA);
        PartitionPlanningContext layerNormContext = planningContext(layerNorm, backendIntentPlan);
        PartitionPlanningContext rmsNormContext = planningContext(rmsNorm, backendIntentPlan);

        String layerNormReason = CudaGpuBackendPartitionCapability.plannerUnsupportedReason(layerNormContext.compiledNode(nodeId(layerNormContext, Operation.OpType.LAYER_NORM)), layerNormContext);
        String rmsNormReason = CudaGpuBackendPartitionCapability.plannerUnsupportedReason(rmsNormContext.compiledNode(nodeId(rmsNormContext, Operation.OpType.RMS_NORM)), rmsNormContext);

        assertEquals("", layerNormReason);
        assertEquals("", rmsNormReason);
    }

    @Test
    void cudaPhaseSeventeenConvAndLossReasonsIncludeMatrixDetail() {
        Tensor input = new Tensor(
                new float[]{
                        1f, 2f, 3f,
                        4f, 5f, 6f,
                        7f, 8f, 9f
                },
                new int[]{1, 1, 3, 3},
                null,
                "cudaPhase17ConvInput",
                DataType.FLOAT32);
        Tensor weight = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 1, 2, 2}, null, "cudaPhase17ConvWeight", DataType.FLOAT32);
        Tensor conv = input.conv2d(weight, Conv2dOptions.defaults());
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(conv, ComputeBackend.GPU_CUDA);
        PartitionPlanningContext convContext = planningContext(conv, backendIntentPlan);
        String convReason = CudaGpuBackendPartitionCapability.plannerUnsupportedReason(convContext.compiledNode(nodeId(convContext, Operation.OpType.CONV2D)), convContext);

        assertContainsAll(convReason,
                "family=CONV_POOL",
                "target=conv2d_resnet_3x3",
                "operation CONV2D is not supported by GPU_CUDA lowering");

        Tensor logits = new Tensor(new float[]{1f, 2f, 3f, 1f, 0f, -1f}, new int[]{2, 3}, null, "cudaPhase17LossLogits", DataType.FLOAT32);
        Tensor targetIndices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "cudaPhase17LossTargets", DataType.INT32);
        Tensor loss = logits.crossEntropyLossFromIndices(targetIndices, 1);
        backendIntentPlan = backendIntentPlan.withBackend(loss, ComputeBackend.GPU_CUDA);
        PartitionPlanningContext lossContext = planningContext(loss, backendIntentPlan);
        String lossReason = CudaGpuBackendPartitionCapability.plannerUnsupportedReason(lossContext.compiledNode(nodeId(lossContext, Operation.OpType.CROSS_ENTROPY_LOSS_INDICES)), lossContext);

        assertContainsAll(lossReason,
                "family=LOSS_ADJACENT",
                "target=transformer_block_hot_path",
                "operation CROSS_ENTROPY_LOSS_INDICES is not supported by GPU_CUDA lowering");
    }

    @Test
    void acceptsLogSoftmaxAsSoftmaxLikeGpuRegion() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "cudaLogSoftmaxInput", DataType.FLOAT32);
        Tensor weight = new Tensor(new float[]{1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f}, new int[]{3, 3}, null, "cudaLogSoftmaxWeight", DataType.FLOAT32);
        Tensor matmul = input.matmul(weight);
        Tensor out = specialLogSoftmax(matmul, 1);
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(matmul, ComputeBackend.GPU_CUDA);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_CUDA);
        PartitionPlanningContext context = planningContext(out, backendIntentPlan);
        int matmulNodeId = nodeId(context, operations.Operation.OpType.MATMUL);
        int logSoftmaxNodeId = nodeId(context, operations.Operation.OpType.LOG_SOFTMAX);
        CudaGpuBackendPartitionCapability adapter = new CudaGpuBackendPartitionCapability();
        var candidate = adapter.createCandidate(
                Set.of(matmulNodeId, logSoftmaxNodeId),
                context,
                Set.of(GraphValueRef.node(logSoftmaxNodeId))
        );
        assertNotNull(candidate);

        CudaGpuPartitionPlan plan = (CudaGpuPartitionPlan) adapter.createPlan(candidate, context);

        assertNotNull(plan);
        List<AcceleratorDagNodeType> types = plan.dagSpec().nodes().stream()
                .map(AcceleratorDagNode::type)
                .toList();
        assertTrue(types.contains(AcceleratorDagNodeType.SOFTMAX));
        assertTrue(types.contains(AcceleratorDagNodeType.LOG));
    }

    @Test
    void supportsSumReductionWithStableCoverageReason() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "cudaStableReductionInput", DataType.FLOAT32);
        Tensor out = input.sum(1);
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_CUDA);
        PartitionPlanningContext context = planningContext(out, backendIntentPlan);
        String reason = CudaGpuBackendPartitionCapability.plannerUnsupportedReason(context.compiledNode(nodeId(context, operations.Operation.OpType.SUM)), context);

        assertEquals("", reason);
    }

    @Test
    void acceptsLayerNormWithStableCoverageReason() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "cudaStableNormInput", DataType.FLOAT32);
        Tensor gamma = new Tensor(new float[]{1f, 1f}, new int[]{2}, null, "cudaStableNormGamma", DataType.FLOAT32);
        Tensor beta = new Tensor(new float[]{0f, 0f}, new int[]{2}, null, "cudaStableNormBeta", DataType.FLOAT32);
        Tensor out = input.layerNorm(gamma, beta, 1.0e-5);
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_CUDA);
        PartitionPlanningContext context = planningContext(out, backendIntentPlan);
        String reason = CudaGpuBackendPartitionCapability.plannerUnsupportedReason(context.compiledNode(nodeId(context, operations.Operation.OpType.LAYER_NORM)), context);

        assertEquals("", reason);
    }

    @Test
    void rejectsCrossEntropyLossWithStableCoverageReason() {
        Tensor logits = new Tensor(new float[]{1f, 2f, 3f, 1f, 0f, -1f}, new int[]{2, 3}, null, "cudaStableLossLogits", DataType.FLOAT32);
        Tensor targetIndices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "cudaStableLossTargets", DataType.INT32);
        Tensor out = logits.crossEntropyLossFromIndices(targetIndices, 1);
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_CUDA);
        PartitionPlanningContext context = planningContext(out, backendIntentPlan);
        String reason = CudaGpuBackendPartitionCapability.plannerUnsupportedReason(context.compiledNode(nodeId(context, operations.Operation.OpType.CROSS_ENTROPY_LOSS_INDICES)), context);

        assertTrue(reason.contains("UNSUPPORTED_INDEX_SEMANTICS"));
    }

    @Test
    void phaseTwentySixIndexFamilyUsesStableCoverageReasons() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "cudaPhase26IndexInput", DataType.FLOAT32);
        Tensor gatherIndices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "cudaPhase26GatherIndices", DataType.INT32);
        Tensor gather = input.gather(gatherIndices, 1);
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(gather, ComputeBackend.GPU_CUDA);
        PartitionPlanningContext gatherContext = planningContext(gather, backendIntentPlan);
        String gatherReason = CudaGpuBackendPartitionCapability.plannerUnsupportedReason(gatherContext.compiledNode(nodeId(gatherContext, Operation.OpType.GATHER)), gatherContext);

        Tensor takeIndices = new Tensor(new int[]{2, 1, 0, 0}, new int[]{2, 2}, null, "cudaPhase26TakeIndices", DataType.INT32);
        Tensor take = input.takeAlongAxis(takeIndices, 1);
        backendIntentPlan = backendIntentPlan.withBackend(take, ComputeBackend.GPU_CUDA);
        PartitionPlanningContext takeContext = planningContext(take, backendIntentPlan);
        String takeReason = CudaGpuBackendPartitionCapability.plannerUnsupportedReason(takeContext.compiledNode(nodeId(takeContext, Operation.OpType.TAKE_ALONG_AXIS)), takeContext);

        Tensor base = new Tensor(new float[]{10f, 20f, 30f, 40f, 50f, 60f}, new int[]{2, 3}, null, "cudaPhase26ScatterBase", DataType.FLOAT32);
        Tensor scatterIndices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "cudaPhase26ScatterIndices", DataType.INT32);
        Tensor src = new Tensor(new float[]{1f, 5f}, new int[]{2}, null, "cudaPhase26ScatterSrc", DataType.FLOAT32);
        Tensor scatter = base.scatterAdd(scatterIndices, src, 1);
        backendIntentPlan = backendIntentPlan.withBackend(scatter, ComputeBackend.GPU_CUDA);
        PartitionPlanningContext scatterContext = planningContext(scatter, backendIntentPlan);
        String scatterReason = CudaGpuBackendPartitionCapability.plannerUnsupportedReason(scatterContext.compiledNode(nodeId(scatterContext, Operation.OpType.SCATTER_ADD)), scatterContext);

        assertContainsAll(gatherReason, "CAPABILITY_MISSING", "operation GATHER", "family=INDEX_SCATTER_GATHER");
        assertContainsAll(takeReason, "CAPABILITY_MISSING", "operation TAKE_ALONG_AXIS", "family=INDEX_SCATTER_GATHER");
        assertContainsAll(scatterReason, "UNSUPPORTED_DUPLICATE_INDEX", "operation SCATTER_ADD", "family=INDEX_SCATTER_GATHER");
    }

    @Test
    void phaseThirtySixIndexGradientOpsKeepStableDuplicateIndexRejections() {
        Tensor indices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "cudaPhase36GradIndices", DataType.INT32);
        Tensor outGrad = new Tensor(new float[]{1f, 2f}, new int[]{2}, null, "cudaPhase36GatherOutGrad", DataType.FLOAT32);
        Tensor gatherGrad = TensorPrimitiveBuilder.binary(
                indices,
                outGrad,
                new int[]{2, 3},
                new gatherGrad(1),
                "cudaPhase36GatherGrad",
                DataType.FLOAT32
        );
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(gatherGrad, ComputeBackend.GPU_CUDA);
        PartitionPlanningContext gatherGradContext = planningContext(gatherGrad, backendIntentPlan);

        Tensor takeIndices = new Tensor(new int[]{2, 2, 0, 0}, new int[]{2, 2}, null, "cudaPhase36TakeGradIndices", DataType.INT32);
        Tensor takeOutGrad = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "cudaPhase36TakeOutGrad", DataType.FLOAT32);
        Tensor takeGrad = TensorPrimitiveBuilder.binary(
                takeIndices,
                takeOutGrad,
                new int[]{2, 3},
                new takeAlongAxisGrad(1),
                "cudaPhase36TakeGrad",
                DataType.FLOAT32
        );
        backendIntentPlan = backendIntentPlan.withBackend(takeGrad, ComputeBackend.GPU_CUDA);
        PartitionPlanningContext takeGradContext = planningContext(takeGrad, backendIntentPlan);

        assertContainsAll(
                CudaGpuBackendPartitionCapability.plannerUnsupportedReason(
                        gatherGradContext.compiledNode(nodeId(gatherGradContext, Operation.OpType.GATHER_GRAD)),
                        gatherGradContext
                ),
                "UNSUPPORTED_DUPLICATE_INDEX",
                "operation GATHER_GRAD",
                "family=INDEX_SCATTER_GATHER",
                "duplicate-index accumulation parity"
        );
        assertContainsAll(
                CudaGpuBackendPartitionCapability.plannerUnsupportedReason(
                        takeGradContext.compiledNode(nodeId(takeGradContext, Operation.OpType.TAKE_ALONG_AXIS_GRAD)),
                        takeGradContext
                ),
                "UNSUPPORTED_DUPLICATE_INDEX",
                "operation TAKE_ALONG_AXIS_GRAD",
                "family=INDEX_SCATTER_GATHER",
                "rank-preserving static bounds checks"
        );
    }

    @Test
    void cudaIndexWriteRejectsDtypeLayoutAndBoundsBeforeDuplicateBlocker() {
        Tensor base = new Tensor(new float[]{10f, 20f, 30f, 40f, 50f, 60f}, new int[]{2, 3}, null, "cuda43ScatterBase", DataType.FLOAT32);
        Tensor src = new Tensor(new float[]{1f, 5f}, new int[]{2}, null, "cuda43ScatterSrc", DataType.FLOAT32);

        Tensor floatIndices = new Tensor(new float[]{2f, 0f}, new int[]{2}, null, "cuda43ScatterFloatIndices", DataType.FLOAT32);
        Tensor dtypeScatter = base.scatterAdd(floatIndices, src, 1);
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(dtypeScatter, ComputeBackend.GPU_CUDA);
        PartitionPlanningContext dtypeContext = planningContext(dtypeScatter, backendIntentPlan);
        assertContainsAll(
                CudaGpuBackendPartitionCapability.plannerUnsupportedReason(
                        dtypeContext.compiledNode(nodeId(dtypeContext, Operation.OpType.SCATTER_ADD)),
                        dtypeContext
                ),
                "UNSUPPORTED_DTYPE",
                "role=INDEX_INPUT",
                "dtype=FLOAT32"
        );

        Tensor oobIndices = new Tensor(new int[]{3, 0}, new int[]{2}, null, "cuda43ScatterOobIndices", DataType.INT32);
        Tensor oobScatter = base.scatterAdd(oobIndices, src, 1);
        backendIntentPlan = backendIntentPlan.withBackend(oobScatter, ComputeBackend.GPU_CUDA);
        PartitionPlanningContext oobContext = planningContext(oobScatter, backendIntentPlan);
        assertContainsAll(
                CudaGpuBackendPartitionCapability.plannerUnsupportedReason(
                        oobContext.compiledNode(nodeId(oobContext, Operation.OpType.SCATTER_ADD)),
                        oobContext
                ),
                "UNSUPPORTED_BOUNDS_CHECK",
                "index 3 is outside axis size 3"
        );

        Tensor layoutBase = new Tensor(new float[]{10f, 40f, 20f, 50f, 30f, 60f}, new int[]{3, 2}, null, "cuda43ScatterLayoutBase", DataType.FLOAT32);
        Tensor nonDenseBase = layoutBase.permute(1, 0);
        Tensor intIndices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "cuda43ScatterIntIndices", DataType.INT32);
        Tensor layoutScatter = nonDenseBase.scatterAdd(intIndices, src, 1);
        backendIntentPlan = backendIntentPlan.withBackend(layoutScatter, ComputeBackend.GPU_CUDA);
        PartitionPlanningContext layoutContext = planningContext(layoutScatter, backendIntentPlan);
        assertContainsAll(
                CudaGpuBackendPartitionCapability.plannerUnsupportedReason(
                        layoutContext.compiledNode(nodeId(layoutContext, Operation.OpType.SCATTER_ADD)),
                        layoutContext
                ),
                "UNSUPPORTED_LAYOUT",
                "SCATTER_ADD inputs require dense layout"
        );
    }

    @Test
    void cudaIndexGradientRejectsBoundsAndUnprovenIndexBeforeDuplicateBlocker() {
        Tensor gatherOobIndices = new Tensor(new int[]{3, 0}, new int[]{2}, null, "cuda43GatherOobIndices", DataType.INT32);
        Tensor gatherOutGrad = new Tensor(new float[]{1f, 2f}, new int[]{2}, null, "cuda43GatherBoundsOutGrad", DataType.FLOAT32);
        Tensor gatherGradOut = TensorPrimitiveBuilder.binary(
                gatherOobIndices,
                gatherOutGrad,
                new int[]{2, 3},
                new gatherGrad(1),
                "cuda43GatherBoundsGrad",
                DataType.FLOAT32
        );
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(gatherGradOut, ComputeBackend.GPU_CUDA);
        PartitionPlanningContext gatherBoundsContext = planningContext(gatherGradOut, backendIntentPlan);
        assertContainsAll(
                CudaGpuBackendPartitionCapability.plannerUnsupportedReason(
                        gatherBoundsContext.compiledNode(nodeId(gatherBoundsContext, Operation.OpType.GATHER_GRAD)),
                        gatherBoundsContext
                ),
                "UNSUPPORTED_BOUNDS_CHECK",
                "index 3 is outside axis size 3"
        );

        Tensor takeOobIndices = new Tensor(new int[]{2, 3, 0, 0}, new int[]{2, 2}, null, "cuda43TakeOobIndices", DataType.INT32);
        Tensor takeOutGrad = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "cuda43TakeBoundsOutGrad", DataType.FLOAT32);
        Tensor takeGradOut = TensorPrimitiveBuilder.binary(
                takeOobIndices,
                takeOutGrad,
                new int[]{2, 3},
                new takeAlongAxisGrad(1),
                "cuda43TakeBoundsGrad",
                DataType.FLOAT32
        );
        backendIntentPlan = backendIntentPlan.withBackend(takeGradOut, ComputeBackend.GPU_CUDA);
        PartitionPlanningContext takeBoundsContext = planningContext(takeGradOut, backendIntentPlan);
        assertContainsAll(
                CudaGpuBackendPartitionCapability.plannerUnsupportedReason(
                        takeBoundsContext.compiledNode(nodeId(takeBoundsContext, Operation.OpType.TAKE_ALONG_AXIS_GRAD)),
                        takeBoundsContext
                ),
                "UNSUPPORTED_BOUNDS_CHECK",
                "index 3 is outside axis size 3"
        );

        Tensor dynamicGatherIndices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "cuda43DynamicGatherIndices", DataType.INT32)
                .reshape(2);
        Tensor dynamicGatherGrad = TensorPrimitiveBuilder.binary(
                dynamicGatherIndices,
                gatherOutGrad,
                new int[]{2, 3},
                new gatherGrad(1),
                "cuda43DynamicGatherGrad",
                DataType.FLOAT32
        );
        backendIntentPlan = backendIntentPlan.withBackend(dynamicGatherGrad, ComputeBackend.GPU_CUDA);
        PartitionPlanningContext dynamicGatherContext = planningContext(dynamicGatherGrad, backendIntentPlan);
        assertContainsAll(
                CudaGpuBackendPartitionCapability.plannerUnsupportedReason(
                        dynamicGatherContext.compiledNode(nodeId(dynamicGatherContext, Operation.OpType.GATHER_GRAD)),
                        dynamicGatherContext
                ),
                "UNSUPPORTED_BOUNDS_CHECK",
                "index bounds require a static INT32 leaf tensor"
        );
    }

    @Test
    void cudaForwardGatherTakeValidateContractBeforeCapabilityMissing() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "cuda41IndexInput", DataType.FLOAT32);
        Tensor gatherIndices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "cuda41GatherIndices", DataType.INT32);
        Tensor gather = input.gather(gatherIndices, 1);
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(gather, ComputeBackend.GPU_CUDA);
        PartitionPlanningContext gatherContext = planningContext(gather, backendIntentPlan);

        Tensor takeIndices = new Tensor(new int[]{2, 1, 0, 0}, new int[]{2, 2}, null, "cuda41TakeIndices", DataType.INT32);
        Tensor take = input.takeAlongAxis(takeIndices, 1);
        backendIntentPlan = backendIntentPlan.withBackend(take, ComputeBackend.GPU_CUDA);
        PartitionPlanningContext takeContext = planningContext(take, backendIntentPlan);

        assertContainsAll(
                CudaGpuBackendPartitionCapability.plannerUnsupportedReason(
                        gatherContext.compiledNode(nodeId(gatherContext, Operation.OpType.GATHER)),
                        gatherContext
                ),
                "CAPABILITY_MISSING",
                "operation GATHER",
                "family=INDEX_SCATTER_GATHER",
                "target=gather_take_small"
        );
        assertContainsAll(
                CudaGpuBackendPartitionCapability.plannerUnsupportedReason(
                        takeContext.compiledNode(nodeId(takeContext, Operation.OpType.TAKE_ALONG_AXIS)),
                        takeContext
                ),
                "CAPABILITY_MISSING",
                "operation TAKE_ALONG_AXIS",
                "family=INDEX_SCATTER_GATHER",
                "target=gather_take_small"
        );
    }

    @Test
    void cudaForwardIndexReportsDtypeLayoutAndBoundsBeforeCapabilityMissing() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "cuda41IndexBadInput", DataType.FLOAT32);
        Tensor f32Indices = new Tensor(new float[]{1f, 0f}, new int[]{2}, null, "cuda41F32GatherIndices", DataType.FLOAT32);
        Tensor dtypeGather = input.gather(f32Indices, 1);
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(dtypeGather, ComputeBackend.GPU_CUDA);
        PartitionPlanningContext dtypeContext = planningContext(dtypeGather, backendIntentPlan);
        assertContainsAll(
                CudaGpuBackendPartitionCapability.plannerUnsupportedReason(
                        dtypeContext.compiledNode(nodeId(dtypeContext, Operation.OpType.GATHER)),
                        dtypeContext
                ),
                "UNSUPPORTED_DTYPE",
                "role=INDEX_INPUT",
                "dtype=FLOAT32"
        );

        Tensor oobIndices = new Tensor(new int[]{3, 0}, new int[]{2}, null, "cuda41OobGatherIndices", DataType.INT32);
        Tensor oobGather = input.gather(oobIndices, 1);
        backendIntentPlan = backendIntentPlan.withBackend(oobGather, ComputeBackend.GPU_CUDA);
        PartitionPlanningContext oobContext = planningContext(oobGather, backendIntentPlan);
        assertContainsAll(
                CudaGpuBackendPartitionCapability.plannerUnsupportedReason(
                        oobContext.compiledNode(nodeId(oobContext, Operation.OpType.GATHER)),
                        oobContext
                ),
                "UNSUPPORTED_BOUNDS_CHECK",
                "index 3 is outside axis size 3"
        );

        Tensor nonDenseValue = input.permute(1, 0);
        Tensor layoutIndices = new Tensor(new int[]{1, 0, 1}, new int[]{3}, null, "cuda41LayoutGatherIndices", DataType.INT32);
        Tensor layoutGather = nonDenseValue.gather(layoutIndices, 1);
        backendIntentPlan = backendIntentPlan.withBackend(layoutGather, ComputeBackend.GPU_CUDA);
        PartitionPlanningContext layoutContext = planningContext(layoutGather, backendIntentPlan);
        assertContainsAll(
                CudaGpuBackendPartitionCapability.plannerUnsupportedReason(
                        layoutContext.compiledNode(nodeId(layoutContext, Operation.OpType.GATHER)),
                        layoutContext
                ),
                "UNSUPPORTED_LAYOUT",
                "inputs require dense value and INT32 index layouts"
        );
    }

    @Test
    void cudaPhaseTwentySevenBoolCompareAndPoolUseStableCoverageReasons() {
        Tensor left = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "cudaPhase27CompareLeft", DataType.FLOAT32);
        Tensor right = new Tensor(new float[]{2f, 2f, 2f, 2f}, new int[]{2, 2}, null, "cudaPhase27CompareRight", DataType.FLOAT32);
        Tensor compare = left.notEqualTo(right);
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(compare, ComputeBackend.GPU_CUDA);
        PartitionPlanningContext compareContext = planningContext(compare, backendIntentPlan);
        String compareReason = CudaGpuBackendPartitionCapability.plannerUnsupportedReason(
                compareContext.compiledNode(nodeId(compareContext, Operation.OpType.NE)),
                compareContext
        );

        Tensor poolInput = new Tensor(new float[]{
                1f, 2f, 3f, 4f,
                5f, 6f, 7f, 8f,
                9f, 10f, 11f, 12f,
                13f, 14f, 15f, 16f
        }, new int[]{1, 1, 4, 4}, null, "cudaPhase27PoolInput", DataType.FLOAT32);
        Tensor maxPool = poolInput.maxPool2d(Pool2dOptions.square(2));
        Tensor avgPool = poolInput.avgPool2d(Pool2dOptions.square(2));
        backendIntentPlan = backendIntentPlan.withBackend(maxPool, ComputeBackend.GPU_CUDA);
        backendIntentPlan = backendIntentPlan.withBackend(avgPool, ComputeBackend.GPU_CUDA);
        PartitionPlanningContext maxPoolContext = planningContext(maxPool, backendIntentPlan);
        PartitionPlanningContext avgPoolContext = planningContext(avgPool, backendIntentPlan);

        String maxPoolReason = CudaGpuBackendPartitionCapability.plannerUnsupportedReason(
                maxPoolContext.compiledNode(nodeId(maxPoolContext, Operation.OpType.MAX_POOL2D)),
                maxPoolContext
        );
        String avgPoolReason = CudaGpuBackendPartitionCapability.plannerUnsupportedReason(
                avgPoolContext.compiledNode(nodeId(avgPoolContext, Operation.OpType.AVG_POOL2D)),
                avgPoolContext
        );

        assertContainsAll(compareReason, "UNSUPPORTED_DTYPE", "operation NE", "family=COMPARE_BOOL", "BOOL output");
        assertContainsAll(maxPoolReason, "CAPABILITY_MISSING", "operation MAX_POOL2D", "family=CONV_POOL", "target=max_pool2d_small");
        assertContainsAll(avgPoolReason, "CAPABILITY_MISSING", "operation AVG_POOL2D", "family=CONV_POOL");
    }

    @Test
    void lowersPureElementwiseGpuCudaRegionToCudaFusedElementwiseGraph() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{4}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{5f, 6f, 7f, 8f}, new int[]{4}, null, "b", DataType.FLOAT32);
        Tensor out = a.add(b).relu().exp();

        List<Tensor> graph = out.topologicalSort();
        List<CompiledNode> compiledNodes = CompiledNodeSnapshotter.snapshot(graph, BackendIntentPlan.empty());
        Partition partition = partition(
                "cuda-fused",
                PartitionTarget.GPU_CUDA,
                List.of(2, 3, 4),
                List.of(0, 1),
                List.of(GraphValueRef.node(4)),
                List.of(GraphValueRef.node(4))
        );
        PlannedRegion region = new DefaultRegionPlanner().planRegion(
                partition,
                new RegionPlanningContext(compiledNodes, FuseConfig.inferenceDefaults())
        );
        CudaGpuPartitionPlan plan = new CudaGpuPartitionPlan(
                4,
                new AcceleratorSubgraphSpec(
                        2,
                        List.of(2, 3, 4),
                        List.of(
                                new AcceleratorSubgraphOp(2, operations.Operation.OpType.ADD),
                                new AcceleratorSubgraphOp(3, operations.Operation.OpType.RELU),
                                new AcceleratorSubgraphOp(4, operations.Operation.OpType.EXP)
                        ),
                        List.of(0, 1),
                        List.of(4)
                ),
                new AcceleratorDagSpec(
                        List.of(
                                new AcceleratorDagInput(0, List.of(4), DataType.FLOAT32),
                                new AcceleratorDagInput(1, List.of(4), DataType.FLOAT32)
                        ),
                        List.of(
                                new AcceleratorDagNode(2, AcceleratorDagNodeType.ADD, AcceleratorDagValueRef.externalInput(0), AcceleratorDagValueRef.externalInput(1), AcceleratorDagValueRef.none(), AcceleratorDagValueRef.none(), 0, 1, 4, 1, 1, 1),
                                new AcceleratorDagNode(3, AcceleratorDagNodeType.RELU, AcceleratorDagValueRef.nodeOutput(0), AcceleratorDagValueRef.none(), AcceleratorDagValueRef.none(), AcceleratorDagValueRef.none(), 0, 1, 4, 1, 1, 1),
                                new AcceleratorDagNode(4, AcceleratorDagNodeType.EXP, AcceleratorDagValueRef.nodeOutput(1), AcceleratorDagValueRef.none(), AcceleratorDagValueRef.none(), AcceleratorDagValueRef.none(), 0, 1, 4, 1, 1, 1)
                        ),
                        List.of(2),
                        List.of(4)
                ),
                12L
        );

        CudaRegionLowerer lowerer = new CudaRegionLowerer();
        LoweringResult result = lowerer.lower(new LoweringRequest(
                region,
                MemoryPlanner.plan(graph),
                new BackendCapabilities(Set.of(ComputeBackend.GPU_CUDA)),
                new LoweringContext(RuntimeConfig.inferenceDefaults(), compiledNodes, CompiledTensorDescriptorBuilder.build(compiledNodes), java.util.Map.of(partition.partitionId(), plan))
        ));

        assertNotNull(result);
        assertNotNull(result.loweredRegion());
        assertEquals(backend.lowering.LoweringFamily.CUDA_GRAPH_REGION, result.loweredRegion().units().getFirst().loweringFamily());
        GpuCompoundLoweringArtifact artifact = result.loweredRegion().units().getFirst().requireArtifact(GpuCompoundLoweringArtifact.class);
        assertTrue(artifact.units().stream().anyMatch(unit -> unit.kind() == ExecutionUnitKind.FUSED_ELEMENTWISE));
    }

    @Test
    void cudaLowersElementwiseSubchainAsRegionInternalFusion() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "cudaSubchainA", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 0f, 0f, 1f, 1f, 1f}, new int[]{3, 2}, null, "cudaSubchainB", DataType.FLOAT32);
        Tensor bias = new Tensor(new float[]{0.25f, -0.5f}, new int[]{2}, null, "cudaSubchainBias", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor add = matmul.add(bias);
        Tensor relu = add.relu();
        Tensor out = relu.exp();
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(matmul, ComputeBackend.GPU_CUDA);
        backendIntentPlan = backendIntentPlan.withBackend(add, ComputeBackend.GPU_CUDA);
        backendIntentPlan = backendIntentPlan.withBackend(relu, ComputeBackend.GPU_CUDA);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_CUDA);
        List<Tensor> graph = out.topologicalSort();
        List<CompiledNode> compiledNodes = CompiledNodeSnapshotter.snapshot(graph, backendIntentPlan);
        PartitionPlanningContext planningContext = new PartitionPlanningContext(
                false,
                compiledNodes,
                CompiledTensorDescriptorBuilder.build(compiledNodes),
                consumers(compiledNodes)
        );
        int matmulNodeId = nodeId(planningContext, Operation.OpType.MATMUL);
        int addNodeId = nodeId(planningContext, Operation.OpType.ADD);
        int reluNodeId = nodeId(planningContext, Operation.OpType.RELU);
        int expNodeId = nodeId(planningContext, Operation.OpType.EXP);
        List<Integer> selectedNodeIds = List.of(matmulNodeId, addNodeId, reluNodeId, expNodeId);
        CudaGpuBackendPartitionCapability adapter = new CudaGpuBackendPartitionCapability();
        var candidate = adapter.createCandidate(
                Set.copyOf(selectedNodeIds),
                planningContext,
                Set.of(GraphValueRef.node(expNodeId))
        );
        assertNotNull(candidate);
        CudaGpuPartitionPlan plan = (CudaGpuPartitionPlan) adapter.createPlan(candidate, planningContext);
        assertNotNull(plan);

        Partition partition = partition(
                "cuda-elementwise-subchain",
                PartitionTarget.GPU_CUDA,
                candidate.orderedNodeIds(),
                candidate.externalInputIds(),
                candidate.outputNodeIds().stream().map(GraphValueRef::node).toList(),
                List.of(GraphValueRef.node(expNodeId))
        );
        PlannedRegion region = new DefaultRegionPlanner().planRegion(
                partition,
                new RegionPlanningContext(compiledNodes, FuseConfig.inferenceDefaults())
        );
        LoweringResult result = new CudaRegionLowerer().lower(new LoweringRequest(
                region,
                MemoryPlanner.plan(graph),
                new BackendCapabilities(Set.of(ComputeBackend.GPU_CUDA)),
                new LoweringContext(RuntimeConfig.inferenceDefaults(), compiledNodes, CompiledTensorDescriptorBuilder.build(compiledNodes), java.util.Map.of(partition.partitionId(), plan))
        ));

        assertNotNull(result);
        assertEquals(backend.lowering.LoweringFamily.CUDA_GRAPH_REGION, result.loweredRegion().units().getFirst().loweringFamily());
        assertTrue(result.loweredRegion().units().getFirst().orderedNodeIds().containsAll(selectedNodeIds));
        var subpattern = plan.manifest().fusedSubpatterns().stream()
                .filter(candidateSubpattern -> candidateSubpattern.patternType() == GpuCompoundPatternType.ELEMENTWISE_CHAIN)
                .findFirst()
                .orElseThrow();
        assertEquals(List.of(addNodeId, reluNodeId, expNodeId), subpattern.originalOperationNodeIds());
        assertEquals(3, subpattern.loweredPrimitiveCount());
    }

    @Test
    void cudaLowersMatmulBiasActivationAsEpilogueSubpattern() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "cudaEpilogueA", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 0f, 0f, 1f, 1f, 1f}, new int[]{3, 2}, null, "cudaEpilogueB", DataType.FLOAT32);
        Tensor bias = new Tensor(new float[]{0.25f, -0.5f}, new int[]{2}, null, "cudaEpilogueBias", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor add = matmul.add(bias);
        Tensor out = add.relu();
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(matmul, ComputeBackend.GPU_CUDA);
        backendIntentPlan = backendIntentPlan.withBackend(add, ComputeBackend.GPU_CUDA);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_CUDA);
        PartitionPlanningContext planningContext = planningContext(out, backendIntentPlan);
        int matmulNodeId = nodeId(planningContext, Operation.OpType.MATMUL);
        int addNodeId = nodeId(planningContext, Operation.OpType.ADD);
        int reluNodeId = nodeId(planningContext, Operation.OpType.RELU);
        List<Integer> selectedNodeIds = List.of(matmulNodeId, addNodeId, reluNodeId);
        CudaGpuBackendPartitionCapability adapter = new CudaGpuBackendPartitionCapability();
        var candidate = adapter.createCandidate(
                Set.copyOf(selectedNodeIds),
                planningContext,
                Set.of(GraphValueRef.node(reluNodeId))
        );
        assertNotNull(candidate);
        CudaGpuPartitionPlan plan = (CudaGpuPartitionPlan) adapter.createPlan(candidate, planningContext);
        assertNotNull(plan);
        var epilogue = plan.manifest().fusedSubpatterns().stream()
                .filter(candidateSubpattern -> candidateSubpattern.patternType() == GpuCompoundPatternType.LINEAR_BIAS_ACTIVATION)
                .findFirst()
                .orElseThrow();

        assertEquals(selectedNodeIds, epilogue.originalOperationNodeIds());
        assertTrue(epilogue.detail().contains("epilogue"));
        assertTrue(epilogue.loweredPrimitiveCount() >= 2);
    }

    @Test
    void cudaRejectsIllegalEpilogueLayoutWithStableReason() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "cudaBadEpilogueA", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 0f, 0f, 1f, 1f, 1f}, new int[]{3, 2}, null, "cudaBadEpilogueB", DataType.FLOAT32);
        Tensor biasBase = new Tensor(new float[]{0.25f, -0.5f, 0.75f, 1f}, new int[]{2, 2}, null, "cudaBadEpilogueBiasBase", DataType.FLOAT32);
        Tensor bias = biasBase.select(0, 1);
        Tensor matmul = a.matmul(b);
        Tensor add = matmul.add(bias);
        Tensor out = add.relu();
        BackendIntentPlan backendIntentPlan = BackendIntentPlan.empty();
        backendIntentPlan = backendIntentPlan.withBackend(matmul, ComputeBackend.GPU_CUDA);
        backendIntentPlan = backendIntentPlan.withBackend(add, ComputeBackend.GPU_CUDA);
        backendIntentPlan = backendIntentPlan.withBackend(out, ComputeBackend.GPU_CUDA);
        PartitionPlanningContext context = planningContext(out, backendIntentPlan);
        String reason = CudaGpuBackendPartitionCapability.plannerUnsupportedReason(context.compiledNode(nodeId(context, Operation.OpType.ADD)), context);

        assertContainsAll(reason,
                "UNSUPPORTED_LAYOUT",
                "family=MATMUL_LINEAR",
                "LINEAR_BIAS_ACTIVATION",
                "GPU_CUDA");
    }

    private static Partition partition(
            String id,
            PartitionTarget target,
            List<Integer> orderedNodeIds,
            List<Integer> externalInputNodeIds,
            List<GraphValueRef> outputValueRefs,
            List<GraphValueRef> requiredMaterialized
    ) {
        List<PartitionValue> values = orderedNodeIds.stream()
                .map(nodeId -> new PartitionValue(GraphValueRef.node(nodeId), nodeId))
                .toList();
        List<PartitionEdge> internalEdges = orderedNodeIds.size() < 2
                ? List.of()
                : java.util.stream.IntStream.range(0, orderedNodeIds.size() - 1)
                .mapToObj(i -> new PartitionEdge(orderedNodeIds.get(i), orderedNodeIds.get(i + 1)))
                .toList();
        return new Partition(
                id,
                target,
                orderedNodeIds,
                values,
                internalEdges,
                externalInputNodeIds,
                outputValueRefs,
                orderedNodeIds.getLast(),
                requiredMaterialized,
                List.of(),
                List.of(PartitionBoundaryReason.NONE),
                orderedNodeIds.size(),
                new planning.partition.cost.AcceleratorPartitionScoreModel.CandidateMetrics(orderedNodeIds.size(), internalEdges.size(), externalInputNodeIds.size(), 0, Math.max(0, orderedNodeIds.size() - 1)),
                PartitionPlannerStrategy.GREEDY_MAX_REGION,
                new PartitionDecisionTrace(
                        PartitionPlannerStrategy.GREEDY_MAX_REGION.name(),
                        target.name(),
                        orderedNodeIds.getLast(),
                        true,
                        "test",
                        orderedNodeIds,
                        orderedNodeIds,
                        List.of(),
                        orderedNodeIds.size(),
                        0.0d,
                        0.0d,
                        0,
                        false,
                        -1
                )
        );
    }

    private static PartitionPlanningContext planningContext(Tensor out) {
        return planningContext(out, BackendIntentPlan.empty());
    }

    private static PartitionPlanningContext planningContext(Tensor out, BackendIntentPlan backendIntentPlan) {
        List<CompiledNode> compiledNodes = CompiledNodeSnapshotter.snapshot(out.topologicalSort(), backendIntentPlan);
        return new PartitionPlanningContext(
                false,
                compiledNodes,
                CompiledTensorDescriptorBuilder.build(compiledNodes),
                consumers(compiledNodes)
        );
    }

    private static java.util.Map<Integer, java.util.List<CompiledNode>> consumers(List<CompiledNode> graph) {
        java.util.Map<Integer, java.util.List<CompiledNode>> consumers = new java.util.HashMap<>();
        for (CompiledNode node : graph) {
            consumers.computeIfAbsent(node.id(), ignored -> new java.util.ArrayList<>());
        }
        for (CompiledNode node : graph) {
            for (int inputId : node.inputIds()) {
                consumers.computeIfAbsent(inputId, ignored -> new java.util.ArrayList<>()).add(node);
            }
        }
        return consumers;
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

    private static Tensor specialSdpa(Tensor query, Tensor key, Tensor value, Tensor mask, double scale) {
        int[] outShape = query.getShapeUnsafe().clone();
        outShape[outShape.length - 1] = value.getShapeUnsafe()[value.getShapeUnsafe().length - 1];
        java.util.ArrayList<Tensor> inputs = new java.util.ArrayList<>();
        inputs.add(query);
        inputs.add(key);
        inputs.add(value);
        if (mask != null) {
            inputs.add(mask);
        }
        return TensorPrimitiveBuilder.nary(
                outShape,
                inputs,
                new operations.linalg.scaledDotProductAttention(scale, mask != null),
                "legacyScaledDotProductAttention",
                query.getDataType()
        );
    }

    private static Tensor causalMask() {
        return new Tensor(new byte[]{
                1, 0,
                1, 1
        }, new int[]{1, 2, 2}, null, "causal_mask", DataType.BOOL);
    }

    private static int nodeId(PartitionPlanningContext context, operations.Operation.OpType opType) {
        return context.compiledNodes().stream()
                .filter(node -> node.operation() != null && node.operation().opType() == opType)
                .map(CompiledNode::id)
                .findFirst()
                .orElseThrow();
    }

    private static void assertContainsAll(String actual, String... expectedSubstrings) {
        for (String expected : expectedSubstrings) {
            assertTrue(actual.contains(expected), () -> "Expected '" + actual + "' to contain '" + expected + "'");
        }
    }

    private record SyntheticFusedOperation() implements Operation {
        @Override
        public OpType opType() {
            return OpType.FUSED;
        }

        @Override
        public OpArityClass arityClass() {
            return OpArityClass.FUSED;
        }

        @Override
        public boolean isFusable() {
            return false;
        }

        @Override
        public OpSemanticFamily semanticFamily() {
            return OpSemanticFamily.FUSED;
        }

        @Override
        public OpComputationalCost computationalCost() {
            return OpComputationalCost.UNKNOWN;
        }

        @Override
        public OpControlTrait controlTrait() {
            return OpControlTrait.UNKNOWN;
        }

        @Override
        public OpResultKind resultKind() {
            return OpResultKind.UNKNOWN;
        }

        @Override
        public String getExpression() {
            return "synthetic_fused";
        }
    }
}
