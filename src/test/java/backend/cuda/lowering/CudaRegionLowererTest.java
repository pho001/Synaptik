package backend.cuda.lowering;

import backend.ComputeBackend;
import backend.accelerator.lowering.GpuCompoundLoweringArtifact;
import backend.accelerator.lowering.GpuCompoundPatternType;
import backend.accelerator.lowering.GpuLoweringCoverageMatrix;
import backend.lowering.BackendCapabilities;
import backend.lowering.LoweringContext;
import backend.lowering.LoweringRequest;
import backend.lowering.LoweringResult;
import config.optimizer.FuseConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledNode;
import graph.execution.trace.PartitionDecisionTrace;
import graph.optimizer.memory.MemoryPlanner;
import graph.optimizer.partition.Partition;
import graph.optimizer.partition.PartitionBoundaryReason;
import graph.optimizer.partition.PartitionEdge;
import graph.optimizer.partition.PartitionPlannerStrategy;
import graph.optimizer.partition.PartitionTarget;
import graph.optimizer.partition.PartitionValue;
import graph.optimizer.partition.PartitionValueRef;
import graph.optimizer.partition.PartitionPlanningContext;
import backend.accelerator.dag.AcceleratorDagInput;
import backend.accelerator.dag.AcceleratorDagNode;
import backend.accelerator.dag.AcceleratorDagNodeType;
import backend.accelerator.dag.AcceleratorDagSpec;
import backend.accelerator.dag.AcceleratorDagValueRef;
import backend.accelerator.dag.AcceleratorSubgraphOp;
import backend.accelerator.dag.AcceleratorSubgraphSpec;
import graph.optimizer.region.DefaultRegionOptimizer;
import graph.optimizer.region.OptimizedRegion;
import graph.optimizer.region.RegionOptimizationContext;
import operations.Operation;
import operations.index.gatherGrad;
import operations.index.takeAlongAxisGrad;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.TensorPrimitiveBuilder;
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
        TensorInternalAccess.setBackend(linear, ComputeBackend.GPU_CUDA);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_CUDA);

        List<Tensor> graph = out.topologicalSort();
        List<CompiledNode> compiledNodes = CompiledNode.snapshot(graph);
        PartitionPlanningContext context = new PartitionPlanningContext(
                RuntimeConfig.inferenceDefaults(),
                false,
                compiledNodes,
                consumers(compiledNodes)
        );
        int linearNodeId = nodeId(context, operations.Operation.OpType.LINEAR);
        int reluNodeId = nodeId(context, operations.Operation.OpType.RELU);
        CudaGpuRegionLegalityAdapter adapter = new CudaGpuRegionLegalityAdapter();
        var candidate = adapter.tryCreateStructuralCandidate(
                Set.of(linearNodeId, reluNodeId),
                context,
                Set.of(PartitionValueRef.ofNode(reluNodeId))
        );
        assertNotNull(candidate);
        CudaGpuPartitionPlan plan = (CudaGpuPartitionPlan) adapter.tryCreatePlan(candidate, context);

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
                candidate.outputNodeIds().stream().map(PartitionValueRef::ofNode).toList(),
                List.of(PartitionValueRef.ofNode(reluNodeId))
        );
        OptimizedRegion region = new DefaultRegionOptimizer().optimize(
                partition,
                new RegionOptimizationContext(compiledNodes, FuseConfig.inferenceDefaults())
        );
        LoweringResult result = new CudaRegionLowerer().lower(new LoweringRequest(
                region,
                MemoryPlanner.plan(graph),
                new BackendCapabilities(Set.of(ComputeBackend.GPU_CUDA)),
                new LoweringContext(RuntimeConfig.inferenceDefaults(), compiledNodes, java.util.Map.of(partition.partitionId(), plan))
        ));

        assertNotNull(result);
        GpuCompoundLoweringArtifact artifact = result.loweredRegion().units().getFirst().requireArtifact(GpuCompoundLoweringArtifact.class);
        assertEquals(GpuCompoundPatternType.LINEAR_BIAS_ACTIVATION, artifact.summary().patternType());
        assertTrue(artifact.summary().orderedNodeIds().containsAll(List.of(linearNodeId, reluNodeId)));
    }

    @Test
    void phaseNineteenCudaLowererKeepsMultiOpRegionAsSingleGraphUnit() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "phase19CudaA", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 0f, 0f, 1f, 1f, 1f}, new int[]{3, 2}, null, "phase19CudaB", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor relu = matmul.relu();
        Tensor out = relu.exp();
        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_CUDA);
        TensorInternalAccess.setBackend(relu, ComputeBackend.GPU_CUDA);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_CUDA);
        List<Tensor> graph = out.topologicalSort();
        List<CompiledNode> compiledNodes = CompiledNode.snapshot(graph);
        PartitionPlanningContext planningContext = new PartitionPlanningContext(
                RuntimeConfig.inferenceDefaults(),
                false,
                compiledNodes,
                consumers(compiledNodes)
        );
        int matmulNodeId = nodeId(planningContext, Operation.OpType.MATMUL);
        int reluNodeId = nodeId(planningContext, Operation.OpType.RELU);
        int expNodeId = nodeId(planningContext, Operation.OpType.EXP);
        List<Integer> selectedNodeIds = List.of(matmulNodeId, reluNodeId, expNodeId);
        CudaGpuRegionLegalityAdapter adapter = new CudaGpuRegionLegalityAdapter();
        var candidate = adapter.tryCreateStructuralCandidate(
                Set.copyOf(selectedNodeIds),
                planningContext,
                Set.of(PartitionValueRef.ofNode(expNodeId))
        );
        assertNotNull(candidate);
        CudaGpuPartitionPlan attachedPlan = (CudaGpuPartitionPlan) adapter.tryCreatePlan(candidate, planningContext);
        assertNotNull(attachedPlan);
        Partition partition = partition(
                "phase19-cuda-multi-op",
                PartitionTarget.GPU_CUDA,
                candidate.orderedNodeIds(),
                candidate.externalInputIds(),
                candidate.outputNodeIds().stream().map(PartitionValueRef::ofNode).toList(),
                List.of(PartitionValueRef.ofNode(expNodeId))
        );
        OptimizedRegion region = new DefaultRegionOptimizer().optimize(
                partition,
                new RegionOptimizationContext(compiledNodes, FuseConfig.inferenceDefaults())
        );

        LoweringResult result = new CudaRegionLowerer().lower(new LoweringRequest(
                region,
                MemoryPlanner.plan(graph),
                new BackendCapabilities(Set.of(ComputeBackend.GPU_CUDA)),
                new LoweringContext(RuntimeConfig.inferenceDefaults(), compiledNodes, java.util.Map.of(partition.partitionId(), attachedPlan))
        ));

        assertNotNull(result);
        assertEquals(1, result.loweredRegion().units().size());
        assertEquals(backend.lowering.LoweringFamily.CUDA_GRAPH_REGION, result.loweredRegion().units().getFirst().loweringFamily());
        assertTrue(attachedPlan.manifest().selectedRegionLength() > 1);
        assertTrue(attachedPlan.manifest().loweredPrimitives().size() > 1);
        assertTrue(result.loweredRegion().units().getFirst().orderedNodeIds().containsAll(selectedNodeIds));
    }

    @Test
    void cudaPlannerSupportMatchesSharedCoverageMatrixForForwardOps() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "cudaMatrixA", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{3, 2}, null, "cudaMatrixB", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor relu = matmul.relu();
        Tensor out = relu.exp();
        Tensor logSoftmax = out.logSoftmax(1);
        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_CUDA);
        TensorInternalAccess.setBackend(relu, ComputeBackend.GPU_CUDA);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_CUDA);
        TensorInternalAccess.setBackend(logSoftmax, ComputeBackend.GPU_CUDA);

        PartitionPlanningContext context = planningContext(logSoftmax);
        for (operations.Operation.OpType opType : List.of(operations.Operation.OpType.MATMUL, operations.Operation.OpType.RELU, operations.Operation.OpType.EXP, operations.Operation.OpType.LOG_SOFTMAX)) {
            assertTrue(GpuLoweringCoverageMatrix.isSupported(ComputeBackend.GPU_CUDA, opType));
            assertEquals("", CudaGpuRegionLegalityAdapter.plannerUnsupportedReason(context.compiledNode(nodeId(context, opType)), context));
        }
    }

    @Test
    void cudaReductionIsPlannerSupported() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "cudaReductionInput", DataType.FLOAT32);
        Tensor out = input.sum(1);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_CUDA);

        PartitionPlanningContext context = planningContext(out);
        String reason = CudaGpuRegionLegalityAdapter.plannerUnsupportedReason(context.compiledNode(nodeId(context, operations.Operation.OpType.SUM)), context);

        assertEquals("", reason);
    }

    @Test
    void cudaSupportedNormalizationUsesSharedCoverageReason() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "cudaNormInput", DataType.FLOAT32);
        Tensor gamma = new Tensor(new float[]{1f, 1f}, new int[]{2}, null, "cudaNormGamma", DataType.FLOAT32);
        Tensor beta = new Tensor(new float[]{0f, 0f}, new int[]{2}, null, "cudaNormBeta", DataType.FLOAT32);
        Tensor out = input.layerNorm(gamma, beta, 1.0e-5);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_CUDA);

        PartitionPlanningContext context = planningContext(out);
        String reason = CudaGpuRegionLegalityAdapter.plannerUnsupportedReason(context.compiledNode(nodeId(context, operations.Operation.OpType.LAYER_NORM)), context);

        assertEquals("", reason);
    }

    @Test
    void cudaDirectForwardSdpaRemainsCapabilityMissingUntilNativeEvidenceExists() {
        Tensor q = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "cudaSdpaQ", DataType.FLOAT32);
        Tensor k = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "cudaSdpaK", DataType.FLOAT32);
        Tensor v = new Tensor(new float[]{10f, 1f, 1f, 10f}, new int[]{1, 2, 2}, null, "cudaSdpaV", DataType.FLOAT32);
        Tensor out = q.scaledDotProductAttention(k, v, AttentionOptions.defaults().withScale(0.5));
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_CUDA);

        PartitionPlanningContext context = planningContext(out);
        String reason = CudaGpuRegionLegalityAdapter.plannerUnsupportedReason(
                context.compiledNode(nodeId(context, Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION)),
                context
        );

        assertTrue(reason.contains("CAPABILITY_MISSING"));
        assertTrue(reason.contains("CUDA direct forward SDPA"));
        assertTrue(reason.contains("target=transformer_block_hot_path"));
        assertFalse(GpuLoweringCoverageMatrix.isSupported(ComputeBackend.GPU_CUDA, Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION));
    }

    @Test
    void cudaMaskedForwardSdpaUsesStableMaskRejectionReason() {
        Tensor q = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "cudaMaskedSdpaQ", DataType.FLOAT32);
        Tensor k = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "cudaMaskedSdpaK", DataType.FLOAT32);
        Tensor v = new Tensor(new float[]{10f, 1f, 1f, 10f}, new int[]{1, 2, 2}, null, "cudaMaskedSdpaV", DataType.FLOAT32);
        Tensor mask = new Tensor(new byte[]{1, 0, 1, 1}, new int[]{1, 2, 2}, null, "cudaMaskedSdpaMask", DataType.BOOL);
        Tensor out = q.scaledDotProductAttention(k, v, mask, AttentionOptions.defaults().withScale(0.5));
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_CUDA);

        PartitionPlanningContext context = planningContext(out);
        String reason = CudaGpuRegionLegalityAdapter.plannerUnsupportedReason(
                context.compiledNode(nodeId(context, Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION)),
                context
        );

        assertTrue(reason.contains("UNSUPPORTED_MASK_SEMANTICS"));
        assertTrue(reason.contains("BOOL mask semantics"));
    }

    @Test
    void cudaForwardSdpaReportsDtypeAndLayoutBeforeCapabilityMissing() {
        Tensor q64 = new Tensor(new double[]{1d, 0d, 0d, 1d}, new int[]{1, 2, 2}, null, "cudaSdpaQ64", DataType.FLOAT64);
        Tensor k64 = new Tensor(new double[]{1d, 0d, 0d, 1d}, new int[]{1, 2, 2}, null, "cudaSdpaK64", DataType.FLOAT64);
        Tensor v64 = new Tensor(new double[]{10d, 1d, 1d, 10d}, new int[]{1, 2, 2}, null, "cudaSdpaV64", DataType.FLOAT64);
        Tensor dtypeOut = q64.scaledDotProductAttention(k64, v64, AttentionOptions.defaults().withScale(0.5));
        TensorInternalAccess.setBackend(dtypeOut, ComputeBackend.GPU_CUDA);
        PartitionPlanningContext dtypeContext = planningContext(dtypeOut);

        String dtypeReason = CudaGpuRegionLegalityAdapter.plannerUnsupportedReason(
                dtypeContext.compiledNode(nodeId(dtypeContext, Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION)),
                dtypeContext
        );
        assertTrue(dtypeReason.contains("UNSUPPORTED_DTYPE"));

        Tensor baseQ = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "cudaSdpaBaseQ", DataType.FLOAT32);
        Tensor qView = baseQ.permute(0, 2, 1);
        Tensor k = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "cudaSdpaDenseK", DataType.FLOAT32);
        Tensor v = new Tensor(new float[]{10f, 1f, 1f, 10f}, new int[]{1, 2, 2}, null, "cudaSdpaDenseV", DataType.FLOAT32);
        Tensor layoutOut = qView.scaledDotProductAttention(k, v, AttentionOptions.defaults().withScale(0.5));
        TensorInternalAccess.setBackend(layoutOut, ComputeBackend.GPU_CUDA);
        PartitionPlanningContext layoutContext = planningContext(layoutOut);

        String layoutReason = CudaGpuRegionLegalityAdapter.plannerUnsupportedReason(
                layoutContext.compiledNode(nodeId(layoutContext, Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION)),
                layoutContext
        );
        assertTrue(layoutReason.contains("UNSUPPORTED_LAYOUT"));
    }

    @Test
    void cudaGpuFusedOpTypeRejectsWithStableCompoundReason() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{4}, null, "cudaFusedInput", DataType.FLOAT32);
        Tensor out = TensorPrimitiveBuilder.unary(input, new SyntheticFusedOperation(), "cudaCpuFusedOp", DataType.FLOAT32);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_CUDA);

        PartitionPlanningContext context = planningContext(out);
        String reason = CudaGpuRegionLegalityAdapter.plannerUnsupportedReason(context.compiledNode(nodeId(context, Operation.OpType.FUSED)), context);

        assertTrue(reason.contains("CPU_FUSED_OPERATION_UNSUPPORTED"));
        assertTrue(reason.contains("operation FUSED is not supported by GPU_CUDA lowering"));
    }

    @Test
    void cudaUnsupportedLossAdjacentUsesSharedUnsupportedReason() {
        Tensor logits = new Tensor(new float[]{1f, 2f, 3f, 1f, 0f, -1f}, new int[]{2, 3}, null, "cudaLossLogits", DataType.FLOAT32);
        Tensor targetIndices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "cudaLossTargets", DataType.INT32);
        Tensor out = logits.crossEntropyLossFromIndices(targetIndices, 1);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_CUDA);

        PartitionPlanningContext context = planningContext(out);
        String reason = CudaGpuRegionLegalityAdapter.plannerUnsupportedReason(context.compiledNode(nodeId(context, operations.Operation.OpType.CROSS_ENTROPY_LOSS_INDICES)), context);

        assertTrue(reason.contains("UNSUPPORTED_INDEX_SEMANTICS"));
        assertTrue(reason.contains("operation CROSS_ENTROPY_LOSS_INDICES is not supported by GPU_CUDA lowering"));
    }

    @Test
    void cudaPhaseSeventeenKeepsDirectNonDenseLayoutRejectionBeforeExecution() {
        Tensor base = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "cudaBase", DataType.FLOAT32);
        Tensor nonDense = base.permute(1, 0);
        Tensor rhs = new Tensor(new float[]{1f, 1f, 1f, 1f, 1f, 1f}, new int[]{3, 2}, null, "cudaRhs", DataType.FLOAT32);
        Tensor out = nonDense.add(rhs);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_CUDA);

        PartitionPlanningContext context = planningContext(out);
        String reason = CudaGpuRegionLegalityAdapter.plannerUnsupportedReason(context.compiledNode(nodeId(context, operations.Operation.OpType.ADD)), context);

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
        TensorInternalAccess.setBackend(sum, ComputeBackend.GPU_CUDA);
        TensorInternalAccess.setBackend(mean, ComputeBackend.GPU_CUDA);
        PartitionPlanningContext sumContext = planningContext(sum);
        PartitionPlanningContext meanContext = planningContext(mean);

        assertEquals("", CudaGpuRegionLegalityAdapter.plannerUnsupportedReason(sumContext.compiledNode(nodeId(sumContext, Operation.OpType.SUM)), sumContext));
        assertEquals("", CudaGpuRegionLegalityAdapter.plannerUnsupportedReason(meanContext.compiledNode(nodeId(meanContext, Operation.OpType.MEAN)), meanContext));

        Tensor normInput = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "cudaPhase17NormInput", DataType.FLOAT32);
        Tensor gamma = new Tensor(new float[]{1f, 1f}, new int[]{2}, null, "cudaPhase17NormGamma", DataType.FLOAT32);
        Tensor beta = new Tensor(new float[]{0f, 0f}, new int[]{2}, null, "cudaPhase17NormBeta", DataType.FLOAT32);
        Tensor layerNorm = normInput.layerNorm(gamma, beta, 1.0e-5);
        Tensor rmsNorm = normInput.rmsNorm(gamma, 1.0e-5);
        TensorInternalAccess.setBackend(layerNorm, ComputeBackend.GPU_CUDA);
        TensorInternalAccess.setBackend(rmsNorm, ComputeBackend.GPU_CUDA);
        PartitionPlanningContext layerNormContext = planningContext(layerNorm);
        PartitionPlanningContext rmsNormContext = planningContext(rmsNorm);

        String layerNormReason = CudaGpuRegionLegalityAdapter.plannerUnsupportedReason(layerNormContext.compiledNode(nodeId(layerNormContext, Operation.OpType.LAYER_NORM)), layerNormContext);
        String rmsNormReason = CudaGpuRegionLegalityAdapter.plannerUnsupportedReason(rmsNormContext.compiledNode(nodeId(rmsNormContext, Operation.OpType.RMS_NORM)), rmsNormContext);

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
        TensorInternalAccess.setBackend(conv, ComputeBackend.GPU_CUDA);

        PartitionPlanningContext convContext = planningContext(conv);
        String convReason = CudaGpuRegionLegalityAdapter.plannerUnsupportedReason(convContext.compiledNode(nodeId(convContext, Operation.OpType.CONV2D)), convContext);

        assertContainsAll(convReason,
                "family=CONV_POOL",
                "target=conv2d_resnet_3x3",
                "operation CONV2D is not supported by GPU_CUDA lowering");

        Tensor logits = new Tensor(new float[]{1f, 2f, 3f, 1f, 0f, -1f}, new int[]{2, 3}, null, "cudaPhase17LossLogits", DataType.FLOAT32);
        Tensor targetIndices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "cudaPhase17LossTargets", DataType.INT32);
        Tensor loss = logits.crossEntropyLossFromIndices(targetIndices, 1);
        TensorInternalAccess.setBackend(loss, ComputeBackend.GPU_CUDA);

        PartitionPlanningContext lossContext = planningContext(loss);
        String lossReason = CudaGpuRegionLegalityAdapter.plannerUnsupportedReason(lossContext.compiledNode(nodeId(lossContext, Operation.OpType.CROSS_ENTROPY_LOSS_INDICES)), lossContext);

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
        Tensor out = matmul.logSoftmax(1);
        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_CUDA);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_CUDA);

        PartitionPlanningContext context = planningContext(out);
        int matmulNodeId = nodeId(context, operations.Operation.OpType.MATMUL);
        int logSoftmaxNodeId = nodeId(context, operations.Operation.OpType.LOG_SOFTMAX);
        CudaGpuRegionLegalityAdapter adapter = new CudaGpuRegionLegalityAdapter();
        var candidate = adapter.tryCreateStructuralCandidate(
                Set.of(matmulNodeId, logSoftmaxNodeId),
                context,
                Set.of(PartitionValueRef.ofNode(logSoftmaxNodeId))
        );
        assertNotNull(candidate);

        CudaGpuPartitionPlan plan = (CudaGpuPartitionPlan) adapter.tryCreatePlan(candidate, context);

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
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_CUDA);

        PartitionPlanningContext context = planningContext(out);
        String reason = CudaGpuRegionLegalityAdapter.plannerUnsupportedReason(context.compiledNode(nodeId(context, operations.Operation.OpType.SUM)), context);

        assertEquals("", reason);
    }

    @Test
    void acceptsLayerNormWithStableCoverageReason() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "cudaStableNormInput", DataType.FLOAT32);
        Tensor gamma = new Tensor(new float[]{1f, 1f}, new int[]{2}, null, "cudaStableNormGamma", DataType.FLOAT32);
        Tensor beta = new Tensor(new float[]{0f, 0f}, new int[]{2}, null, "cudaStableNormBeta", DataType.FLOAT32);
        Tensor out = input.layerNorm(gamma, beta, 1.0e-5);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_CUDA);

        PartitionPlanningContext context = planningContext(out);
        String reason = CudaGpuRegionLegalityAdapter.plannerUnsupportedReason(context.compiledNode(nodeId(context, operations.Operation.OpType.LAYER_NORM)), context);

        assertEquals("", reason);
    }

    @Test
    void rejectsCrossEntropyLossWithStableCoverageReason() {
        Tensor logits = new Tensor(new float[]{1f, 2f, 3f, 1f, 0f, -1f}, new int[]{2, 3}, null, "cudaStableLossLogits", DataType.FLOAT32);
        Tensor targetIndices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "cudaStableLossTargets", DataType.INT32);
        Tensor out = logits.crossEntropyLossFromIndices(targetIndices, 1);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_CUDA);

        PartitionPlanningContext context = planningContext(out);
        String reason = CudaGpuRegionLegalityAdapter.plannerUnsupportedReason(context.compiledNode(nodeId(context, operations.Operation.OpType.CROSS_ENTROPY_LOSS_INDICES)), context);

        assertTrue(reason.contains("UNSUPPORTED_INDEX_SEMANTICS"));
    }

    @Test
    void phaseTwentySixIndexFamilyUsesStableCoverageReasons() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "cudaPhase26IndexInput", DataType.FLOAT32);
        Tensor gatherIndices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "cudaPhase26GatherIndices", DataType.INT32);
        Tensor gather = input.gather(gatherIndices, 1);
        TensorInternalAccess.setBackend(gather, ComputeBackend.GPU_CUDA);
        PartitionPlanningContext gatherContext = planningContext(gather);
        String gatherReason = CudaGpuRegionLegalityAdapter.plannerUnsupportedReason(gatherContext.compiledNode(nodeId(gatherContext, Operation.OpType.GATHER)), gatherContext);

        Tensor takeIndices = new Tensor(new int[]{2, 1, 0, 0}, new int[]{2, 2}, null, "cudaPhase26TakeIndices", DataType.INT32);
        Tensor take = input.takeAlongAxis(takeIndices, 1);
        TensorInternalAccess.setBackend(take, ComputeBackend.GPU_CUDA);
        PartitionPlanningContext takeContext = planningContext(take);
        String takeReason = CudaGpuRegionLegalityAdapter.plannerUnsupportedReason(takeContext.compiledNode(nodeId(takeContext, Operation.OpType.TAKE_ALONG_AXIS)), takeContext);

        Tensor base = new Tensor(new float[]{10f, 20f, 30f, 40f, 50f, 60f}, new int[]{2, 3}, null, "cudaPhase26ScatterBase", DataType.FLOAT32);
        Tensor scatterIndices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "cudaPhase26ScatterIndices", DataType.INT32);
        Tensor src = new Tensor(new float[]{1f, 5f}, new int[]{2}, null, "cudaPhase26ScatterSrc", DataType.FLOAT32);
        Tensor scatter = base.scatterAdd(scatterIndices, src, 1);
        TensorInternalAccess.setBackend(scatter, ComputeBackend.GPU_CUDA);
        PartitionPlanningContext scatterContext = planningContext(scatter);
        String scatterReason = CudaGpuRegionLegalityAdapter.plannerUnsupportedReason(scatterContext.compiledNode(nodeId(scatterContext, Operation.OpType.SCATTER_ADD)), scatterContext);

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
        TensorInternalAccess.setBackend(gatherGrad, ComputeBackend.GPU_CUDA);
        PartitionPlanningContext gatherGradContext = planningContext(gatherGrad);

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
        TensorInternalAccess.setBackend(takeGrad, ComputeBackend.GPU_CUDA);
        PartitionPlanningContext takeGradContext = planningContext(takeGrad);

        assertContainsAll(
                CudaGpuRegionLegalityAdapter.plannerUnsupportedReason(
                        gatherGradContext.compiledNode(nodeId(gatherGradContext, Operation.OpType.GATHER_GRAD)),
                        gatherGradContext
                ),
                "UNSUPPORTED_DUPLICATE_INDEX",
                "operation GATHER_GRAD",
                "family=INDEX_SCATTER_GATHER",
                "duplicate-index accumulation parity"
        );
        assertContainsAll(
                CudaGpuRegionLegalityAdapter.plannerUnsupportedReason(
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
    void cudaPhaseTwentySevenBoolCompareAndPoolUseStableCoverageReasons() {
        Tensor left = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "cudaPhase27CompareLeft", DataType.FLOAT32);
        Tensor right = new Tensor(new float[]{2f, 2f, 2f, 2f}, new int[]{2, 2}, null, "cudaPhase27CompareRight", DataType.FLOAT32);
        Tensor compare = left.notEqualTo(right);
        TensorInternalAccess.setBackend(compare, ComputeBackend.GPU_CUDA);
        PartitionPlanningContext compareContext = planningContext(compare);
        String compareReason = CudaGpuRegionLegalityAdapter.plannerUnsupportedReason(
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
        TensorInternalAccess.setBackend(maxPool, ComputeBackend.GPU_CUDA);
        TensorInternalAccess.setBackend(avgPool, ComputeBackend.GPU_CUDA);
        PartitionPlanningContext maxPoolContext = planningContext(maxPool);
        PartitionPlanningContext avgPoolContext = planningContext(avgPool);

        String maxPoolReason = CudaGpuRegionLegalityAdapter.plannerUnsupportedReason(
                maxPoolContext.compiledNode(nodeId(maxPoolContext, Operation.OpType.MAX_POOL2D)),
                maxPoolContext
        );
        String avgPoolReason = CudaGpuRegionLegalityAdapter.plannerUnsupportedReason(
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
        List<CompiledNode> compiledNodes = CompiledNode.snapshot(graph);
        Partition partition = partition(
                "cuda-fused",
                PartitionTarget.GPU_CUDA,
                List.of(2, 3, 4),
                List.of(0, 1),
                List.of(PartitionValueRef.ofNode(4)),
                List.of(PartitionValueRef.ofNode(4))
        );
        OptimizedRegion region = new DefaultRegionOptimizer().optimize(
                partition,
                new RegionOptimizationContext(compiledNodes, FuseConfig.inferenceDefaults())
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
                new LoweringContext(RuntimeConfig.inferenceDefaults(), compiledNodes, java.util.Map.of(partition.partitionId(), plan))
        ));

        assertNotNull(result);
        assertNotNull(result.loweredRegion());
        assertEquals(backend.lowering.LoweringFamily.CUDA_FUSED_ELEMENTWISE_GRAPH, result.loweredRegion().units().getFirst().loweringFamily());
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
        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_CUDA);
        TensorInternalAccess.setBackend(add, ComputeBackend.GPU_CUDA);
        TensorInternalAccess.setBackend(relu, ComputeBackend.GPU_CUDA);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_CUDA);

        List<Tensor> graph = out.topologicalSort();
        List<CompiledNode> compiledNodes = CompiledNode.snapshot(graph);
        PartitionPlanningContext planningContext = new PartitionPlanningContext(
                RuntimeConfig.inferenceDefaults(),
                false,
                compiledNodes,
                consumers(compiledNodes)
        );
        int matmulNodeId = nodeId(planningContext, Operation.OpType.MATMUL);
        int addNodeId = nodeId(planningContext, Operation.OpType.ADD);
        int reluNodeId = nodeId(planningContext, Operation.OpType.RELU);
        int expNodeId = nodeId(planningContext, Operation.OpType.EXP);
        List<Integer> selectedNodeIds = List.of(matmulNodeId, addNodeId, reluNodeId, expNodeId);
        CudaGpuRegionLegalityAdapter adapter = new CudaGpuRegionLegalityAdapter();
        var candidate = adapter.tryCreateStructuralCandidate(
                Set.copyOf(selectedNodeIds),
                planningContext,
                Set.of(PartitionValueRef.ofNode(expNodeId))
        );
        assertNotNull(candidate);
        CudaGpuPartitionPlan plan = (CudaGpuPartitionPlan) adapter.tryCreatePlan(candidate, planningContext);
        assertNotNull(plan);

        Partition partition = partition(
                "cuda-elementwise-subchain",
                PartitionTarget.GPU_CUDA,
                candidate.orderedNodeIds(),
                candidate.externalInputIds(),
                candidate.outputNodeIds().stream().map(PartitionValueRef::ofNode).toList(),
                List.of(PartitionValueRef.ofNode(expNodeId))
        );
        OptimizedRegion region = new DefaultRegionOptimizer().optimize(
                partition,
                new RegionOptimizationContext(compiledNodes, FuseConfig.inferenceDefaults())
        );
        LoweringResult result = new CudaRegionLowerer().lower(new LoweringRequest(
                region,
                MemoryPlanner.plan(graph),
                new BackendCapabilities(Set.of(ComputeBackend.GPU_CUDA)),
                new LoweringContext(RuntimeConfig.inferenceDefaults(), compiledNodes, java.util.Map.of(partition.partitionId(), plan))
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
        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_CUDA);
        TensorInternalAccess.setBackend(add, ComputeBackend.GPU_CUDA);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_CUDA);

        PartitionPlanningContext planningContext = planningContext(out);
        int matmulNodeId = nodeId(planningContext, Operation.OpType.MATMUL);
        int addNodeId = nodeId(planningContext, Operation.OpType.ADD);
        int reluNodeId = nodeId(planningContext, Operation.OpType.RELU);
        List<Integer> selectedNodeIds = List.of(matmulNodeId, addNodeId, reluNodeId);
        CudaGpuRegionLegalityAdapter adapter = new CudaGpuRegionLegalityAdapter();
        var candidate = adapter.tryCreateStructuralCandidate(
                Set.copyOf(selectedNodeIds),
                planningContext,
                Set.of(PartitionValueRef.ofNode(reluNodeId))
        );
        assertNotNull(candidate);
        CudaGpuPartitionPlan plan = (CudaGpuPartitionPlan) adapter.tryCreatePlan(candidate, planningContext);
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
        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_CUDA);
        TensorInternalAccess.setBackend(add, ComputeBackend.GPU_CUDA);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_CUDA);

        PartitionPlanningContext context = planningContext(out);
        String reason = CudaGpuRegionLegalityAdapter.plannerUnsupportedReason(context.compiledNode(nodeId(context, Operation.OpType.ADD)), context);

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
            List<PartitionValueRef> outputValueRefs,
            List<PartitionValueRef> requiredMaterialized
    ) {
        List<PartitionValue> values = orderedNodeIds.stream()
                .map(nodeId -> new PartitionValue(PartitionValueRef.ofNode(nodeId), nodeId))
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
                new graph.optimizer.partition.cost.AcceleratorPartitionScoreModel.CandidateMetrics(orderedNodeIds.size(), internalEdges.size(), externalInputNodeIds.size(), 0, Math.max(0, orderedNodeIds.size() - 1)),
                PartitionPlannerStrategy.GREEDY_MAX_REGION,
                new PartitionDecisionTrace(
                        PartitionPlannerStrategy.GREEDY_MAX_REGION,
                        target,
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
        List<CompiledNode> compiledNodes = CompiledNode.snapshot(out.topologicalSort());
        return new PartitionPlanningContext(
                RuntimeConfig.inferenceDefaults(),
                false,
                compiledNodes,
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
        public String getExpression() {
            return "synthetic_fused";
        }
    }
}
