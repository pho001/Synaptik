package backend.metal.lowering;

import backend.ComputeBackend;
import backend.accelerator.dag.AcceleratorDagNodeType;
import backend.accelerator.dag.AcceleratorSubgraphSpec;
import backend.accelerator.lowering.AcceleratorSubgraphLowerer;
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
import graph.optimizer.partition.PartitionCandidate;
import graph.optimizer.partition.PartitionEdge;
import graph.optimizer.partition.PartitionPlannerStrategy;
import graph.optimizer.partition.PartitionPlanningContext;
import graph.optimizer.partition.PartitionTarget;
import graph.optimizer.partition.PartitionValue;
import graph.optimizer.partition.PartitionValueRef;
import graph.optimizer.region.DefaultRegionOptimizer;
import graph.optimizer.region.ExecutionUnitKind;
import graph.optimizer.region.OptimizedRegion;
import graph.optimizer.region.RegionOptimizationContext;
import operations.Operation;
import operations.elementwise.where.where;
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
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class MetalRegionLowererTest {
    @Test
    void metalLowersLinearBiasReluAsLinearBiasActivationCompoundRegion() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "metalLinearInput", DataType.FLOAT32);
        Tensor weight = new Tensor(new float[]{
                1f, 0f, 0f, 1f,
                0f, 1f, 1f, 0f,
                1f, 1f, 0f, 0f
        }, new int[]{3, 4}, null, "metalLinearWeight", DataType.FLOAT32);
        Tensor bias = new Tensor(new float[]{0.5f, -0.5f, 1f, -1f}, new int[]{4}, null, "metalLinearBias", DataType.FLOAT32);
        Tensor linear = input.linear(weight, bias);
        Tensor out = linear.relu();
        TensorInternalAccess.setBackend(linear, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        List<Tensor> graph = out.topologicalSort();
        List<CompiledNode> compiledNodes = CompiledNode.snapshot(graph);
        PartitionPlanningContext context = new PartitionPlanningContext(
                RuntimeConfig.inferenceDefaults(),
                false,
                compiledNodes,
                consumers(compiledNodes)
        );
        int linearNodeId = nodeId(context, Operation.OpType.LINEAR);
        int reluNodeId = nodeId(context, Operation.OpType.RELU);
        MetalRegionLegalityAdapter adapter = new MetalRegionLegalityAdapter();
        PartitionCandidate candidate = adapter.tryCreateStructuralCandidate(
                Set.of(linearNodeId, reluNodeId),
                context,
                Set.of(PartitionValueRef.ofNode(reluNodeId))
        );
        assertNotNull(candidate);
        MetalPartitionPlan attachedPlan = (MetalPartitionPlan) adapter.tryCreatePlan(candidate, context);

        assertNotNull(attachedPlan);
        assertEquals(GpuCompoundPatternType.LINEAR_BIAS_ACTIVATION, attachedPlan.lowering().compoundSummary().patternType());
        assertTrue(attachedPlan.lowering().compoundSummary().supported());
        assertTrue(attachedPlan.nodeIds().containsAll(List.of(linearNodeId, reluNodeId)));
        assertEquals(ComputeBackend.GPU_METAL, attachedPlan.manifest().backend());
        assertTrue(attachedPlan.manifest().selectedRegionLength() >= attachedPlan.nodeIds().size());
        assertTrue(attachedPlan.lowering().dagSpec().nodes().stream().anyMatch(node -> node.type() == AcceleratorDagNodeType.LINEAR));
        assertTrue(attachedPlan.lowering().dagSpec().nodes().stream().anyMatch(node -> node.type() == AcceleratorDagNodeType.RELU));
        assertTrue(attachedPlan.matMulSpec().biasInputNodeId() >= 0);
        assertTrue(attachedPlan.matMulSpec().postOps().stream()
                .anyMatch(postOp -> postOp.type() == backend.accelerator.dag.AcceleratorPostOpType.RELU));

        Partition partition = new Partition(
                "metal-linear-bias-activation",
                PartitionTarget.GPU_METAL,
                candidate.orderedNodeIds(),
                candidate.orderedNodeIds().stream().map(id -> new PartitionValue(PartitionValueRef.ofNode(id), id)).toList(),
                List.of(new PartitionEdge(linearNodeId, reluNodeId)),
                candidate.externalInputIds(),
                candidate.outputNodeIds().stream().map(PartitionValueRef::ofNode).toList(),
                candidate.anchorNodeId(),
                List.of(PartitionValueRef.ofNode(reluNodeId)),
                List.of(),
                List.of(PartitionBoundaryReason.NONE),
                attachedPlan.estimatedWork(),
                new graph.optimizer.partition.cost.AcceleratorPartitionScoreModel.CandidateMetrics(candidate.orderedNodeIds().size(), 1, candidate.externalInputIds().size(), 0, 1),
                PartitionPlannerStrategy.GREEDY_MAX_REGION,
                new PartitionDecisionTrace(
                        PartitionPlannerStrategy.GREEDY_MAX_REGION,
                        PartitionTarget.GPU_METAL,
                        candidate.anchorNodeId(),
                        true,
                        "test",
                        candidate.orderedNodeIds(),
                        candidate.orderedNodeIds(),
                        List.of("LINEAR", "RELU"),
                        attachedPlan.estimatedWork(),
                        0.0d,
                        0.0d,
                        0,
                        false,
                        -1
                )
        );
        OptimizedRegion region = new DefaultRegionOptimizer().optimize(partition, new RegionOptimizationContext(compiledNodes, FuseConfig.inferenceDefaults()));

        LoweringResult result = new MetalRegionLowerer().lower(new LoweringRequest(
                region,
                MemoryPlanner.plan(graph),
                new BackendCapabilities(Set.of(ComputeBackend.GPU_METAL)),
                new LoweringContext(RuntimeConfig.inferenceDefaults(), compiledNodes, java.util.Map.of(partition.partitionId(), attachedPlan))
        ));

        assertNotNull(result);
        GpuCompoundLoweringArtifact artifact = result.loweredRegion().units().getFirst().requireArtifact(GpuCompoundLoweringArtifact.class);
        assertEquals(GpuCompoundPatternType.LINEAR_BIAS_ACTIVATION, artifact.summary().patternType());
        assertTrue(artifact.summary().orderedNodeIds().containsAll(List.of(linearNodeId, reluNodeId)));
    }

    @Test
    void phaseNineteenMetalLowererKeepsMultiOpRegionAsSingleGraphUnit() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "phase19MetalA", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 0f, 0f, 1f, 1f, 1f}, new int[]{3, 2}, null, "phase19MetalB", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor relu = matmul.relu();
        Tensor out = relu.exp();
        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(relu, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);
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
        MetalRegionLegalityAdapter adapter = new MetalRegionLegalityAdapter();
        PartitionCandidate candidate = adapter.tryCreateStructuralCandidate(
                Set.copyOf(selectedNodeIds),
                planningContext,
                Set.of(PartitionValueRef.ofNode(expNodeId))
        );
        assertNotNull(candidate);
        MetalPartitionPlan attachedPlan = (MetalPartitionPlan) adapter.tryCreatePlan(candidate, planningContext);
        assertNotNull(attachedPlan);
        Partition partition = partition(
                "phase19-metal-multi-op",
                PartitionTarget.GPU_METAL,
                candidate.orderedNodeIds(),
                candidate.externalInputIds(),
                candidate.outputNodeIds().stream().map(PartitionValueRef::ofNode).toList(),
                List.of(PartitionValueRef.ofNode(expNodeId))
        );
        OptimizedRegion region = new DefaultRegionOptimizer().optimize(
                partition,
                new RegionOptimizationContext(compiledNodes, FuseConfig.inferenceDefaults())
        );

        LoweringResult result = new MetalRegionLowerer().lower(new LoweringRequest(
                region,
                MemoryPlanner.plan(graph),
                new BackendCapabilities(Set.of(ComputeBackend.GPU_METAL)),
                new LoweringContext(RuntimeConfig.inferenceDefaults(), compiledNodes, java.util.Map.of(partition.partitionId(), attachedPlan))
        ));

        assertNotNull(result);
        assertEquals(1, result.loweredRegion().units().size());
        assertEquals(backend.lowering.LoweringFamily.METAL_GRAPH_REGION, result.loweredRegion().units().getFirst().loweringFamily());
        assertTrue(attachedPlan.manifest().selectedRegionLength() > 1);
        assertTrue(attachedPlan.manifest().loweredPrimitives().size() > 1);
        assertTrue(result.loweredRegion().units().getFirst().orderedNodeIds().containsAll(selectedNodeIds));
    }

    @Test
    void metalPlannerSupportMatchesSharedCoverageMatrixForForwardOps() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "metalMatrixA", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{3, 2}, null, "metalMatrixB", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor relu = matmul.relu();
        Tensor out = relu.exp();
        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(relu, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PartitionPlanningContext context = planningContext(out);
        for (Operation.OpType opType : List.of(Operation.OpType.MATMUL, Operation.OpType.RELU, Operation.OpType.EXP)) {
            assertTrue(GpuLoweringCoverageMatrix.isSupported(ComputeBackend.GPU_METAL, opType));
            assertEquals("", MetalPartitionSupport.plannerUnsupportedReason(context.compiledNode(nodeId(context, opType)), context));
        }
    }

    @Test
    void metalReductionIsPlannerSupported() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "metalReductionInput", DataType.FLOAT32);
        Tensor out = input.sum(1);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PartitionPlanningContext context = planningContext(out);
        String reason = MetalPartitionSupport.plannerUnsupportedReason(context.compiledNode(nodeId(context, Operation.OpType.SUM)), context);

        assertEquals("", reason);
    }

    @Test
    void metalSupportedNormalizationUsesSharedCoverageReason() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "metalNormInput", DataType.FLOAT32);
        Tensor gamma = new Tensor(new float[]{1f, 1f}, new int[]{2}, null, "metalNormGamma", DataType.FLOAT32);
        Tensor beta = new Tensor(new float[]{0f, 0f}, new int[]{2}, null, "metalNormBeta", DataType.FLOAT32);
        Tensor out = input.layerNorm(gamma, beta, 1.0e-5);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PartitionPlanningContext context = planningContext(out);
        String reason = MetalPartitionSupport.plannerUnsupportedReason(context.compiledNode(nodeId(context, Operation.OpType.LAYER_NORM)), context);

        assertEquals("", reason);
    }

    @Test
    void metalGpuFusedOpTypeRejectsWithStableCompoundReason() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{4}, null, "metalFusedInput", DataType.FLOAT32);
        Tensor out = TensorPrimitiveBuilder.unary(input, new SyntheticFusedOperation(), "metalCpuFusedOp", DataType.FLOAT32);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PartitionPlanningContext context = planningContext(out);
        String reason = MetalPartitionSupport.plannerUnsupportedReason(context.compiledNode(nodeId(context, Operation.OpType.FUSED)), context);

        assertTrue(reason.contains("CPU_FUSED_OPERATION_UNSUPPORTED"));
        assertTrue(reason.contains("operation FUSED is not supported by GPU_METAL lowering"));
    }

    @Test
    void metalUnsupportedLossAdjacentUsesSharedUnsupportedReason() {
        Tensor logits = new Tensor(new float[]{1f, 2f, 3f, 1f, 0f, -1f}, new int[]{2, 3}, null, "metalLossLogits", DataType.FLOAT32);
        Tensor targetIndices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "metalLossTargets", DataType.INT32);
        Tensor out = logits.crossEntropyLossFromIndices(targetIndices, 1);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PartitionPlanningContext context = planningContext(out);
        String reason = MetalPartitionSupport.plannerUnsupportedReason(context.compiledNode(nodeId(context, Operation.OpType.CROSS_ENTROPY_LOSS_INDICES)), context);

        assertTrue(reason.contains("UNSUPPORTED_INDEX_SEMANTICS"));
        assertTrue(reason.contains("operation CROSS_ENTROPY_LOSS_INDICES is not supported by GPU_METAL lowering"));
    }

    @Test
    void metalPhaseSeventeenReductionAndNormReasonsIncludeMatrixDetail() {
        Tensor reductionInput = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "metalPhase17ReductionInput", DataType.FLOAT32);
        Tensor sum = reductionInput.sum(1);
        Tensor mean = reductionInput.mean(1);
        TensorInternalAccess.setBackend(sum, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(mean, ComputeBackend.GPU_METAL);
        PartitionPlanningContext sumContext = planningContext(sum);
        PartitionPlanningContext meanContext = planningContext(mean);

        assertEquals("", MetalPartitionSupport.plannerUnsupportedReason(sumContext.compiledNode(nodeId(sumContext, Operation.OpType.SUM)), sumContext));
        assertEquals("", MetalPartitionSupport.plannerUnsupportedReason(meanContext.compiledNode(nodeId(meanContext, Operation.OpType.MEAN)), meanContext));

        Tensor normInput = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "metalPhase17NormInput", DataType.FLOAT32);
        Tensor gamma = new Tensor(new float[]{1f, 1f}, new int[]{2}, null, "metalPhase17NormGamma", DataType.FLOAT32);
        Tensor beta = new Tensor(new float[]{0f, 0f}, new int[]{2}, null, "metalPhase17NormBeta", DataType.FLOAT32);
        Tensor layerNorm = normInput.layerNorm(gamma, beta, 1.0e-5);
        Tensor rmsNorm = normInput.rmsNorm(gamma, 1.0e-5);
        TensorInternalAccess.setBackend(layerNorm, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(rmsNorm, ComputeBackend.GPU_METAL);
        PartitionPlanningContext layerNormContext = planningContext(layerNorm);
        PartitionPlanningContext rmsNormContext = planningContext(rmsNorm);

        String layerNormReason = MetalPartitionSupport.plannerUnsupportedReason(layerNormContext.compiledNode(nodeId(layerNormContext, Operation.OpType.LAYER_NORM)), layerNormContext);
        String rmsNormReason = MetalPartitionSupport.plannerUnsupportedReason(rmsNormContext.compiledNode(nodeId(rmsNormContext, Operation.OpType.RMS_NORM)), rmsNormContext);

        assertEquals("", layerNormReason);
        assertEquals("", rmsNormReason);
    }

    @Test
    void metalPhaseSeventeenConvAndLossReasonsIncludeMatrixDetail() {
        Tensor input = new Tensor(
                new float[]{
                        1f, 2f, 3f,
                        4f, 5f, 6f,
                        7f, 8f, 9f
                },
                new int[]{1, 1, 3, 3},
                null,
                "metalPhase17ConvInput",
                DataType.FLOAT32);
        Tensor weight = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 1, 2, 2}, null, "metalPhase17ConvWeight", DataType.FLOAT32);
        Tensor conv = input.conv2d(weight, Conv2dOptions.defaults());
        TensorInternalAccess.setBackend(conv, ComputeBackend.GPU_METAL);

        PartitionPlanningContext convContext = planningContext(conv);
        String convReason = MetalPartitionSupport.plannerUnsupportedReason(convContext.compiledNode(nodeId(convContext, Operation.OpType.CONV2D)), convContext);

        assertContainsAll(convReason,
                "family=CONV_POOL",
                "target=conv2d_resnet_3x3",
                "operation CONV2D is not supported by GPU_METAL lowering");

        Tensor logits = new Tensor(new float[]{1f, 2f, 3f, 1f, 0f, -1f}, new int[]{2, 3}, null, "metalPhase17LossLogits", DataType.FLOAT32);
        Tensor targetIndices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "metalPhase17LossTargets", DataType.INT32);
        Tensor loss = logits.crossEntropyLossFromIndices(targetIndices, 1);
        TensorInternalAccess.setBackend(loss, ComputeBackend.GPU_METAL);

        PartitionPlanningContext lossContext = planningContext(loss);
        String lossReason = MetalPartitionSupport.plannerUnsupportedReason(lossContext.compiledNode(nodeId(lossContext, Operation.OpType.CROSS_ENTROPY_LOSS_INDICES)), lossContext);

        assertContainsAll(lossReason,
                "family=LOSS_ADJACENT",
                "target=transformer_block_hot_path",
                "operation CROSS_ENTROPY_LOSS_INDICES is not supported by GPU_METAL lowering");
    }

    @Test
    void acceptsLogSoftmaxAsSoftmaxLikeGpuRegion() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "metalLogSoftmaxInput", DataType.FLOAT32);
        Tensor weight = new Tensor(new float[]{1f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 1f}, new int[]{3, 3}, null, "metalLogSoftmaxWeight", DataType.FLOAT32);
        Tensor matmul = input.matmul(weight);
        Tensor out = matmul.logSoftmax(1);
        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PartitionPlanningContext context = planningContext(out);
        int matmulNodeId = nodeId(context, Operation.OpType.MATMUL);
        int logSoftmaxNodeId = nodeId(context, Operation.OpType.LOG_SOFTMAX);
        MetalRegionLegalityAdapter adapter = new MetalRegionLegalityAdapter();
        PartitionCandidate candidate = adapter.tryCreateStructuralCandidate(
                Set.of(matmulNodeId, logSoftmaxNodeId),
                context,
                Set.of(PartitionValueRef.ofNode(logSoftmaxNodeId))
        );
        assertNotNull(candidate);

        MetalPartitionPlan plan = (MetalPartitionPlan) adapter.tryCreatePlan(candidate, context);

        assertNotNull(plan);
        List<AcceleratorDagNodeType> types = plan.lowering().dagSpec().nodes().stream()
                .map(backend.accelerator.dag.AcceleratorDagNode::type)
                .toList();
        assertTrue(types.contains(AcceleratorDagNodeType.SOFTMAX));
        assertTrue(types.contains(AcceleratorDagNodeType.LOG));
    }

    @Test
    void supportsSumReductionWithStableCoverageReason() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "metalStableReductionInput", DataType.FLOAT32);
        Tensor out = input.sum(1);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PartitionPlanningContext context = planningContext(out);
        String reason = MetalPartitionSupport.plannerUnsupportedReason(context.compiledNode(nodeId(context, Operation.OpType.SUM)), context);

        assertEquals("", reason);
    }

    @Test
    void acceptsLayerNormWithStableCoverageReason() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "metalStableNormInput", DataType.FLOAT32);
        Tensor gamma = new Tensor(new float[]{1f, 1f}, new int[]{2}, null, "metalStableNormGamma", DataType.FLOAT32);
        Tensor beta = new Tensor(new float[]{0f, 0f}, new int[]{2}, null, "metalStableNormBeta", DataType.FLOAT32);
        Tensor out = input.layerNorm(gamma, beta, 1.0e-5);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PartitionPlanningContext context = planningContext(out);
        String reason = MetalPartitionSupport.plannerUnsupportedReason(context.compiledNode(nodeId(context, Operation.OpType.LAYER_NORM)), context);

        assertEquals("", reason);
    }

    @Test
    void rejectsCrossEntropyLossWithStableCoverageReason() {
        Tensor logits = new Tensor(new float[]{1f, 2f, 3f, 1f, 0f, -1f}, new int[]{2, 3}, null, "metalStableLossLogits", DataType.FLOAT32);
        Tensor targetIndices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "metalStableLossTargets", DataType.INT32);
        Tensor out = logits.crossEntropyLossFromIndices(targetIndices, 1);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PartitionPlanningContext context = planningContext(out);
        String reason = MetalPartitionSupport.plannerUnsupportedReason(context.compiledNode(nodeId(context, Operation.OpType.CROSS_ENTROPY_LOSS_INDICES)), context);

        assertTrue(reason.contains("UNSUPPORTED_INDEX_SEMANTICS"));
    }

    @Test
    void phaseThirtyTwoIndexFamilySupportsForwardGatherTakeAndKeepsScatterRejected() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "metalPhase26IndexInput", DataType.FLOAT32);
        Tensor gatherIndices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "metalPhase26GatherIndices", DataType.INT32);
        Tensor gather = input.gather(gatherIndices, 1);
        TensorInternalAccess.setBackend(gather, ComputeBackend.GPU_METAL);
        PartitionPlanningContext gatherContext = planningContext(gather);
        String gatherReason = MetalPartitionSupport.plannerUnsupportedReason(gatherContext.compiledNode(nodeId(gatherContext, Operation.OpType.GATHER)), gatherContext);

        Tensor takeIndices = new Tensor(new int[]{2, 1, 0, 0}, new int[]{2, 2}, null, "metalPhase26TakeIndices", DataType.INT32);
        Tensor take = input.takeAlongAxis(takeIndices, 1);
        TensorInternalAccess.setBackend(take, ComputeBackend.GPU_METAL);
        PartitionPlanningContext takeContext = planningContext(take);
        String takeReason = MetalPartitionSupport.plannerUnsupportedReason(takeContext.compiledNode(nodeId(takeContext, Operation.OpType.TAKE_ALONG_AXIS)), takeContext);

        Tensor base = new Tensor(new float[]{10f, 20f, 30f, 40f, 50f, 60f}, new int[]{2, 3}, null, "metalPhase26ScatterBase", DataType.FLOAT32);
        Tensor scatterIndices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "metalPhase26ScatterIndices", DataType.INT32);
        Tensor src = new Tensor(new float[]{1f, 5f}, new int[]{2}, null, "metalPhase26ScatterSrc", DataType.FLOAT32);
        Tensor scatter = base.scatterAdd(scatterIndices, src, 1);
        TensorInternalAccess.setBackend(scatter, ComputeBackend.GPU_METAL);
        PartitionPlanningContext scatterContext = planningContext(scatter);
        String scatterReason = MetalPartitionSupport.plannerUnsupportedReason(scatterContext.compiledNode(nodeId(scatterContext, Operation.OpType.SCATTER_ADD)), scatterContext);

        assertEquals("", gatherReason);
        assertEquals("", takeReason);
        assertTrue(planFor(gather, Operation.OpType.GATHER).lowering().dagSpec().nodes().stream()
                .anyMatch(node -> node.type() == AcceleratorDagNodeType.GATHER && node.scalarValueBits() == 1));
        assertTrue(planFor(take, Operation.OpType.TAKE_ALONG_AXIS).lowering().dagSpec().nodes().stream()
                .anyMatch(node -> node.type() == AcceleratorDagNodeType.TAKE_ALONG_AXIS && node.scalarValueBits() == 1));
        assertContainsAll(scatterReason, "UNSUPPORTED_DUPLICATE_INDEX", "operation SCATTER_ADD", "family=INDEX_SCATTER_GATHER");
    }

    @Test
    void phaseThirtyTwoIndexFamilyRejectsUnprovenBoundsDtypeAndLayout() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "metalPhase32RejectInput", DataType.FLOAT32);

        Tensor oobIndices = new Tensor(new int[]{3, 0}, new int[]{2}, null, "metalPhase32OobIndices", DataType.INT32);
        Tensor oobGather = input.gather(oobIndices, 1);
        TensorInternalAccess.setBackend(oobGather, ComputeBackend.GPU_METAL);
        PartitionPlanningContext oobContext = planningContext(oobGather);
        assertContainsAll(
                MetalPartitionSupport.plannerUnsupportedReason(oobContext.compiledNode(nodeId(oobContext, Operation.OpType.GATHER)), oobContext),
                "UNSUPPORTED_BOUNDS_CHECK",
                "outside axis size 3"
        );

        Tensor f32Indices = new Tensor(new float[]{2f, 0f}, new int[]{2}, null, "metalPhase32F32Indices", DataType.FLOAT32);
        Tensor dtypeGather = input.gather(f32Indices, 1);
        TensorInternalAccess.setBackend(dtypeGather, ComputeBackend.GPU_METAL);
        PartitionPlanningContext dtypeContext = planningContext(dtypeGather);
        assertContainsAll(
                MetalPartitionSupport.plannerUnsupportedReason(dtypeContext.compiledNode(nodeId(dtypeContext, Operation.OpType.GATHER)), dtypeContext),
                "UNSUPPORTED_DTYPE",
                "index input requires INT32"
        );

        Tensor nonDenseValue = input.permute(1, 0);
        Tensor valueLayoutIndices = new Tensor(new int[]{1, 0}, new int[]{2}, null, "metalPhase32ValueLayoutIndices", DataType.INT32);
        Tensor valueLayoutGather = nonDenseValue.gather(valueLayoutIndices, 0);
        TensorInternalAccess.setBackend(valueLayoutGather, ComputeBackend.GPU_METAL);
        PartitionPlanningContext valueLayoutContext = planningContext(valueLayoutGather);
        assertContainsAll(
                MetalPartitionSupport.plannerUnsupportedReason(valueLayoutContext.compiledNode(nodeId(valueLayoutContext, Operation.OpType.GATHER)), valueLayoutContext),
                "UNSUPPORTED_LAYOUT",
                "inputs require dense layout"
        );

        Tensor denseInput = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "metalPhase32DenseInput", DataType.FLOAT32);
        Tensor indexBase = new Tensor(new int[]{1, 0, 0, 1}, new int[]{2, 2}, null, "metalPhase32IndexBase", DataType.INT32);
        Tensor nonDenseIndices = indexBase.permute(1, 0);
        Tensor indexLayoutTake = denseInput.takeAlongAxis(nonDenseIndices, 1);
        TensorInternalAccess.setBackend(indexLayoutTake, ComputeBackend.GPU_METAL);
        PartitionPlanningContext indexLayoutContext = planningContext(indexLayoutTake);
        assertContainsAll(
                MetalPartitionSupport.plannerUnsupportedReason(indexLayoutContext.compiledNode(nodeId(indexLayoutContext, Operation.OpType.TAKE_ALONG_AXIS)), indexLayoutContext),
                "UNSUPPORTED_LAYOUT",
                "inputs require dense layout"
        );
    }

    @Test
    void metalPhaseTwentySevenBoolCompareAndPoolUseStableCoverageReasons() {
        Tensor left = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "metalPhase27CompareLeft", DataType.FLOAT32);
        Tensor right = new Tensor(new float[]{2f, 2f, 2f, 2f}, new int[]{2, 2}, null, "metalPhase27CompareRight", DataType.FLOAT32);
        Tensor compare = left.greaterOrEqual(right);
        TensorInternalAccess.setBackend(compare, ComputeBackend.GPU_METAL);
        PartitionPlanningContext compareContext = planningContext(compare);
        String compareReason = MetalPartitionSupport.plannerUnsupportedReason(
                compareContext.compiledNode(nodeId(compareContext, Operation.OpType.GE)),
                compareContext
        );

        Tensor poolInput = new Tensor(new float[]{
                1f, 2f, 3f, 4f,
                5f, 6f, 7f, 8f,
                9f, 10f, 11f, 12f,
                13f, 14f, 15f, 16f
        }, new int[]{1, 1, 4, 4}, null, "metalPhase27PoolInput", DataType.FLOAT32);
        Tensor maxPool = poolInput.maxPool2d(Pool2dOptions.square(2));
        Tensor avgPool = poolInput.avgPool2d(Pool2dOptions.square(2));
        TensorInternalAccess.setBackend(maxPool, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(avgPool, ComputeBackend.GPU_METAL);
        PartitionPlanningContext maxPoolContext = planningContext(maxPool);
        PartitionPlanningContext avgPoolContext = planningContext(avgPool);

        String maxPoolReason = MetalPartitionSupport.plannerUnsupportedReason(
                maxPoolContext.compiledNode(nodeId(maxPoolContext, Operation.OpType.MAX_POOL2D)),
                maxPoolContext
        );
        String avgPoolReason = MetalPartitionSupport.plannerUnsupportedReason(
                avgPoolContext.compiledNode(nodeId(avgPoolContext, Operation.OpType.AVG_POOL2D)),
                avgPoolContext
        );

        assertEquals("", compareReason);
        assertContainsAll(maxPoolReason, "CAPABILITY_MISSING", "operation MAX_POOL2D", "family=CONV_POOL", "target=max_pool2d_small");
        assertContainsAll(avgPoolReason, "CAPABILITY_MISSING", "operation AVG_POOL2D", "family=CONV_POOL");
    }

    @Test
    void boolCompareDagContractLowersWithPlannerAdmission() {
        assertEquals(41, AcceleratorDagNodeType.GT.abiCode());
        assertEquals(42, AcceleratorDagNodeType.GE.abiCode());
        assertEquals(43, AcceleratorDagNodeType.LT.abiCode());
        assertEquals(44, AcceleratorDagNodeType.LE.abiCode());
        assertEquals(45, AcceleratorDagNodeType.EQ.abiCode());
        assertEquals(46, AcceleratorDagNodeType.NE.abiCode());
        assertEquals(47, AcceleratorDagNodeType.LOGICAL_AND.abiCode());
        assertEquals(48, AcceleratorDagNodeType.LOGICAL_OR.abiCode());
        assertEquals(49, AcceleratorDagNodeType.LOGICAL_NOT.abiCode());
        assertEquals(50, AcceleratorDagNodeType.REDUCE_ALL.abiCode());
        assertEquals(51, AcceleratorDagNodeType.REDUCE_ANY.abiCode());

        Tensor left = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "metalPhase31BoolLeft", DataType.FLOAT32);
        Tensor right = new Tensor(new float[]{2f, 2f, 2f, 2f}, new int[]{2, 2}, null, "metalPhase31BoolRight", DataType.FLOAT32);
        Tensor compare = left.greaterOrEqual(right);
        TensorInternalAccess.setBackend(compare, ComputeBackend.GPU_METAL);

        PartitionPlanningContext context = planningContext(compare);
        int compareNodeId = nodeId(context, Operation.OpType.GE);
        CompiledNode compareNode = context.compiledNode(compareNodeId);
        var lowered = new AcceleratorSubgraphLowerer().tryLower(
                ComputeBackend.GPU_METAL,
                new AcceleratorSubgraphSpec(
                        compareNodeId,
                        List.of(compareNodeId),
                        List.of(new backend.accelerator.dag.AcceleratorSubgraphOp(compareNodeId, Operation.OpType.GE)),
                        compareNode.inputIds(),
                        List.of(compareNodeId)
                ),
                context
        );

        assertNotNull(lowered);
        assertEquals(AcceleratorDagNodeType.GE, lowered.dagSpec().nodes().getFirst().type());
        assertEquals(DataType.BOOL, lowered.dagSpec().nodes().getFirst().outputDataType());
        assertTrue(lowered.dagSpec().externalInputs().stream().allMatch(input -> input.dataType() == DataType.FLOAT32));
        assertEquals(
                "SUPPORTED",
                GpuLoweringCoverageMatrix.entryFor(ComputeBackend.GPU_METAL, Operation.OpType.GE).reason().name()
        );
        assertNotNull(new MetalRegionLegalityAdapter().tryCreateStructuralCandidate(
                Set.of(compareNodeId),
                context,
                Set.of(PartitionValueRef.ofNode(compareNodeId))
        ));
    }

    @Test
    void boolLogicalAndReductionDagContractsLowerForMetal() {
        Tensor left = new Tensor(new byte[]{1, 0, 1, 0}, new int[]{2, 2}, null, "metalPhase31BoolLeft", DataType.BOOL);
        Tensor right = new Tensor(new byte[]{1, 1, 0, 0}, new int[]{2, 2}, null, "metalPhase31BoolRight", DataType.BOOL);
        Tensor logical = left.logicalAnd(right);
        Tensor reduced = logical.any(1, true);
        TensorInternalAccess.setBackend(logical, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(reduced, ComputeBackend.GPU_METAL);

        PartitionPlanningContext context = planningContext(reduced);
        int logicalNodeId = nodeId(context, Operation.OpType.LOGICAL_AND);
        int reduceNodeId = nodeId(context, Operation.OpType.REDUCE_ANY);
        MetalRegionLegalityAdapter adapter = new MetalRegionLegalityAdapter();
        PartitionCandidate candidate = adapter.tryCreateStructuralCandidate(
                Set.of(logicalNodeId, reduceNodeId),
                context,
                Set.of(PartitionValueRef.ofNode(reduceNodeId))
        );

        assertNotNull(candidate);
        MetalPartitionPlan plan = (MetalPartitionPlan) adapter.tryCreatePlan(candidate, context);
        assertNotNull(plan);
        List<AcceleratorDagNodeType> types = plan.lowering().dagSpec().nodes().stream()
                .map(backend.accelerator.dag.AcceleratorDagNode::type)
                .toList();
        assertTrue(types.contains(AcceleratorDagNodeType.LOGICAL_AND));
        assertTrue(types.contains(AcceleratorDagNodeType.REDUCE_ANY));
        assertTrue(plan.lowering().dagSpec().nodes().stream().allMatch(node -> node.outputDataType() == DataType.BOOL));
    }

    @Test
    void lowersGpuMetalRegionToMetalGraphRegion() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{3, 2}, null, "b", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor out = matmul.relu();
        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        List<Tensor> graph = out.topologicalSort();
        List<CompiledNode> compiledNodes = CompiledNode.snapshot(graph);
        graph.optimizer.partition.PartitionPlanningContext planningContext = new graph.optimizer.partition.PartitionPlanningContext(
                RuntimeConfig.inferenceDefaults(),
                false,
                compiledNodes,
                consumers(compiledNodes)
        );
        MetalRegionLegalityAdapter adapter = new MetalRegionLegalityAdapter();
        PartitionCandidate candidate = adapter.tryCreateStructuralCandidate(
                Set.of(2, 3),
                planningContext,
                Set.of(PartitionValueRef.ofNode(3))
        );
        var attachedPlan = adapter.tryCreatePlan(candidate, planningContext);
        Partition partition = new Partition(
                "metal-partition",
                PartitionTarget.GPU_METAL,
                candidate.orderedNodeIds(),
                candidate.orderedNodeIds().stream().map(id -> new PartitionValue(PartitionValueRef.ofNode(id), id)).toList(),
                List.of(new PartitionEdge(2, 3)),
                candidate.externalInputIds(),
                candidate.outputNodeIds().stream().map(PartitionValueRef::ofNode).toList(),
                candidate.anchorNodeId(),
                List.of(PartitionValueRef.ofNode(3)),
                List.of(),
                List.of(PartitionBoundaryReason.NONE),
                attachedPlan.estimatedWork(),
                new graph.optimizer.partition.cost.AcceleratorPartitionScoreModel.CandidateMetrics(candidate.orderedNodeIds().size(), 1, candidate.externalInputIds().size(), 0, 1),
                PartitionPlannerStrategy.GREEDY_MAX_REGION,
                new PartitionDecisionTrace(
                        PartitionPlannerStrategy.GREEDY_MAX_REGION,
                        PartitionTarget.GPU_METAL,
                        candidate.anchorNodeId(),
                        true,
                        "test",
                        candidate.orderedNodeIds(),
                        candidate.orderedNodeIds(),
                        List.of("MATMUL", "RELU"),
                        attachedPlan.estimatedWork(),
                        0.0d,
                        0.0d,
                        0,
                        false,
                        -1
                )
        );
        OptimizedRegion region = new DefaultRegionOptimizer().optimize(partition, new RegionOptimizationContext(compiledNodes, FuseConfig.inferenceDefaults()));

        MetalRegionLowerer lowerer = new MetalRegionLowerer();
        LoweringResult result = lowerer.lower(new LoweringRequest(
                region,
                MemoryPlanner.plan(graph),
                new BackendCapabilities(Set.of(ComputeBackend.GPU_METAL)),
                new LoweringContext(RuntimeConfig.inferenceDefaults(), compiledNodes, java.util.Map.of(partition.partitionId(), attachedPlan))
        ));

        assertNotNull(result);
        assertNotNull(result.loweredRegion());
        assertEquals(backend.lowering.LoweringFamily.METAL_GRAPH_REGION, result.loweredRegion().units().getFirst().loweringFamily());
    }

    @Test
    void lowersPureElementwiseGpuMetalRegionToMetalFusedElementwiseGraph() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{4}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{5f, 6f, 7f, 8f}, new int[]{4}, null, "b", DataType.FLOAT32);
        Tensor add = a.add(b);
        Tensor out = add.relu().exp();
        TensorInternalAccess.setBackend(add, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out.getPrevTensors().getFirst(), ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        List<Tensor> graph = out.topologicalSort();
        List<CompiledNode> compiledNodes = CompiledNode.snapshot(graph);
        graph.optimizer.partition.PartitionPlanningContext planningContext = new graph.optimizer.partition.PartitionPlanningContext(
                RuntimeConfig.inferenceDefaults(),
                false,
                compiledNodes,
                consumers(compiledNodes)
        );
        MetalRegionLegalityAdapter adapter = new MetalRegionLegalityAdapter();
        PartitionCandidate candidate = adapter.tryCreateStructuralCandidate(
                Set.of(2, 3, 4),
                planningContext,
                Set.of(PartitionValueRef.ofNode(4))
        );
        assertNotNull(candidate);
        var attachedPlan = adapter.tryCreatePlan(candidate, planningContext);
        assertNotNull(attachedPlan);

        Partition partition = new Partition(
                "metal-elementwise-partition",
                PartitionTarget.GPU_METAL,
                candidate.orderedNodeIds(),
                candidate.orderedNodeIds().stream().map(id -> new PartitionValue(PartitionValueRef.ofNode(id), id)).toList(),
                List.of(new PartitionEdge(2, 3), new PartitionEdge(3, 4)),
                candidate.externalInputIds(),
                candidate.outputNodeIds().stream().map(PartitionValueRef::ofNode).toList(),
                candidate.anchorNodeId(),
                List.of(PartitionValueRef.ofNode(4)),
                List.of(),
                List.of(PartitionBoundaryReason.NONE),
                attachedPlan.estimatedWork(),
                new graph.optimizer.partition.cost.AcceleratorPartitionScoreModel.CandidateMetrics(candidate.orderedNodeIds().size(), 2, candidate.externalInputIds().size(), 0, 2),
                PartitionPlannerStrategy.GREEDY_MAX_REGION,
                new PartitionDecisionTrace(
                        PartitionPlannerStrategy.GREEDY_MAX_REGION,
                        PartitionTarget.GPU_METAL,
                        candidate.anchorNodeId(),
                        true,
                        "test",
                        candidate.orderedNodeIds(),
                        candidate.orderedNodeIds(),
                        List.of("ADD", "RELU", "EXP"),
                        attachedPlan.estimatedWork(),
                        0.0d,
                        0.0d,
                        0,
                        false,
                        -1
                )
        );
        OptimizedRegion region = new DefaultRegionOptimizer().optimize(
                partition,
                new RegionOptimizationContext(compiledNodes, FuseConfig.inferenceDefaults())
        );

        assertEquals(1, region.executionUnits().size());
        assertEquals(ExecutionUnitKind.FUSED_ELEMENTWISE, region.executionUnits().getFirst().kind());

        MetalRegionLowerer lowerer = new MetalRegionLowerer();
        LoweringResult result = lowerer.lower(new LoweringRequest(
                region,
                MemoryPlanner.plan(graph),
                new BackendCapabilities(Set.of(ComputeBackend.GPU_METAL)),
                new LoweringContext(RuntimeConfig.inferenceDefaults(), compiledNodes, java.util.Map.of(partition.partitionId(), attachedPlan))
        ));

        assertNotNull(result);
        assertNotNull(result.loweredRegion());
        assertEquals(backend.lowering.LoweringFamily.METAL_FUSED_ELEMENTWISE_GRAPH, result.loweredRegion().units().getFirst().loweringFamily());
    }

    @Test
    void metalLowersElementwiseSubchainAsRegionInternalFusion() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "metalSubchainA", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 0f, 0f, 1f, 1f, 1f}, new int[]{3, 2}, null, "metalSubchainB", DataType.FLOAT32);
        Tensor bias = new Tensor(new float[]{0.25f, -0.5f}, new int[]{2}, null, "metalSubchainBias", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor add = matmul.add(bias);
        Tensor relu = add.relu();
        Tensor out = relu.exp();
        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(add, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(relu, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

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
        MetalRegionLegalityAdapter adapter = new MetalRegionLegalityAdapter();
        PartitionCandidate candidate = adapter.tryCreateStructuralCandidate(
                Set.copyOf(selectedNodeIds),
                planningContext,
                Set.of(PartitionValueRef.ofNode(expNodeId))
        );
        assertNotNull(candidate);
        MetalPartitionPlan attachedPlan = (MetalPartitionPlan) adapter.tryCreatePlan(candidate, planningContext);
        assertNotNull(attachedPlan);

        Partition partition = partition(
                "metal-elementwise-subchain",
                PartitionTarget.GPU_METAL,
                candidate.orderedNodeIds(),
                candidate.externalInputIds(),
                candidate.outputNodeIds().stream().map(PartitionValueRef::ofNode).toList(),
                List.of(PartitionValueRef.ofNode(expNodeId))
        );
        OptimizedRegion region = new DefaultRegionOptimizer().optimize(
                partition,
                new RegionOptimizationContext(compiledNodes, FuseConfig.inferenceDefaults())
        );
        LoweringResult result = new MetalRegionLowerer().lower(new LoweringRequest(
                region,
                MemoryPlanner.plan(graph),
                new BackendCapabilities(Set.of(ComputeBackend.GPU_METAL)),
                new LoweringContext(RuntimeConfig.inferenceDefaults(), compiledNodes, java.util.Map.of(partition.partitionId(), attachedPlan))
        ));

        assertNotNull(result);
        assertEquals(backend.lowering.LoweringFamily.METAL_GRAPH_REGION, result.loweredRegion().units().getFirst().loweringFamily());
        assertTrue(result.loweredRegion().units().getFirst().orderedNodeIds().containsAll(selectedNodeIds));
        var subpattern = attachedPlan.manifest().fusedSubpatterns().stream()
                .filter(candidateSubpattern -> candidateSubpattern.patternType() == GpuCompoundPatternType.ELEMENTWISE_CHAIN)
                .findFirst()
                .orElseThrow();
        assertEquals(List.of(addNodeId, reluNodeId, expNodeId), subpattern.originalOperationNodeIds());
        assertEquals(3, subpattern.loweredPrimitiveCount());
    }

    @Test
    void metalLowersMatmulBiasActivationAsEpilogueSubpattern() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "metalEpilogueA", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 0f, 0f, 1f, 1f, 1f}, new int[]{3, 2}, null, "metalEpilogueB", DataType.FLOAT32);
        Tensor bias = new Tensor(new float[]{0.25f, -0.5f}, new int[]{2}, null, "metalEpilogueBias", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor add = matmul.add(bias);
        Tensor out = add.relu();
        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(add, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

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
        List<Integer> selectedNodeIds = List.of(matmulNodeId, addNodeId, reluNodeId);
        MetalRegionLegalityAdapter adapter = new MetalRegionLegalityAdapter();
        PartitionCandidate candidate = adapter.tryCreateStructuralCandidate(
                Set.copyOf(selectedNodeIds),
                planningContext,
                Set.of(PartitionValueRef.ofNode(reluNodeId))
        );
        assertNotNull(candidate);
        MetalPartitionPlan plan = (MetalPartitionPlan) adapter.tryCreatePlan(candidate, planningContext);
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
    void metalRejectsIllegalEpilogueLayoutWithStableReason() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "metalBadEpilogueA", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 0f, 0f, 1f, 1f, 1f}, new int[]{3, 2}, null, "metalBadEpilogueB", DataType.FLOAT32);
        Tensor biasBase = new Tensor(new float[]{0.25f, -0.5f, 0.75f, 1f}, new int[]{2, 2}, null, "metalBadEpilogueBiasBase", DataType.FLOAT32);
        Tensor bias = biasBase.select(0, 1);
        Tensor matmul = a.matmul(b);
        Tensor add = matmul.add(bias);
        Tensor out = add.relu();
        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(add, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PartitionPlanningContext context = planningContext(out);
        String reason = MetalPartitionSupport.plannerUnsupportedReason(context.compiledNode(nodeId(context, Operation.OpType.ADD)), context);

        assertContainsAll(reason,
                "UNSUPPORTED_LAYOUT",
                "family=MATMUL_LINEAR",
                "LINEAR_BIAS_ACTIVATION",
                "GPU_METAL");
    }

    @Test
    void rejectsFloat64MetalCandidateBeforeLowering() {
        Tensor a = new Tensor(new double[]{1d, 2d, 3d, 4d, 5d, 6d}, new int[]{2, 3}, null, "a64", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{1d, 2d, 3d, 4d, 5d, 6d}, new int[]{3, 2}, null, "b64", DataType.FLOAT64);
        Tensor matmul = a.matmul(b);
        Tensor out = matmul.relu();
        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PartitionPlanningContext context = planningContext(out);
        assertNull(new MetalRegionLegalityAdapter().tryCreateStructuralCandidate(
                operationNodeIds(context),
                context,
                Set.of(PartitionValueRef.ofNode(nodeId(context, Operation.OpType.RELU)))
        ));
    }

    @Test
    void acceptsScopedBfloat16MetalCandidateBeforeLowering() {
        Tensor a = new Tensor(new double[]{1d, 2d, 3d, 4d, 5d, 6d}, new int[]{2, 3}, null, "abf16", DataType.BFLOAT16);
        Tensor b = new Tensor(new double[]{1d, 2d, 3d, 4d, 5d, 6d}, new int[]{3, 2}, null, "bbf16", DataType.BFLOAT16);
        Tensor matmul = a.matmul(b);
        Tensor out = matmul.relu();
        TensorInternalAccess.setBackend(matmul, ComputeBackend.GPU_METAL);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PartitionPlanningContext context = planningContext(out);
        MetalRegionLegalityAdapter adapter = new MetalRegionLegalityAdapter();
        PartitionCandidate candidate = adapter.tryCreateStructuralCandidate(
                operationNodeIds(context),
                context,
                Set.of(PartitionValueRef.ofNode(nodeId(context, Operation.OpType.RELU)))
        );

        assertNotNull(candidate);
        MetalPartitionPlan plan = (MetalPartitionPlan) adapter.tryCreatePlan(candidate, context);
        assertNotNull(plan);
        assertTrue(plan.lowering().dagSpec().externalInputs().stream().allMatch(input -> input.dataType() == DataType.BFLOAT16));
        assertTrue(plan.lowering().dagSpec().nodes().stream().allMatch(node -> node.outputDataType() == DataType.BFLOAT16));
    }

    @Test
    void rejectsInt32MetalLayoutCandidateBeforeLowering() {
        Tensor index = new Tensor(new int[]{1, 2, 3, 4}, new int[]{4}, null, "index", DataType.INT32);
        Tensor out = index.contiguous();
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PartitionPlanningContext context = planningContext(out);
        assertNull(new MetalRegionLegalityAdapter().tryCreateStructuralCandidate(
                operationNodeIds(context),
                context,
                Set.of(PartitionValueRef.ofNode(nodeId(context, Operation.OpType.CONTIGUOUS)))
        ));
    }

    @Test
    void acceptsWhereWithBoolPredicateAndFloat32Branches() {
        Tensor mask = new Tensor(new byte[]{1, 0}, new int[]{2}, null, "mask", DataType.BOOL);
        Tensor x = new Tensor(new float[]{1f, 2f}, new int[]{2}, null, "x", DataType.FLOAT32);
        Tensor y = new Tensor(new float[]{3f, 4f}, new int[]{2}, null, "y", DataType.FLOAT32);
        Tensor out = Tensor.where(mask, x, y);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PartitionPlanningContext context = planningContext(out);
        MetalRegionLegalityAdapter adapter = new MetalRegionLegalityAdapter();
        PartitionCandidate candidate = adapter.tryCreateStructuralCandidate(
                operationNodeIds(context),
                context,
                Set.of(PartitionValueRef.ofNode(nodeId(context, Operation.OpType.WHERE)))
        );

        assertNotNull(candidate);
        assertNotNull(adapter.tryCreatePlan(candidate, context));
    }

    @Test
    void rejectsWhereWithBoolValueBranch() {
        Tensor mask = new Tensor(new byte[]{1, 0}, new int[]{2}, null, "mask", DataType.BOOL);
        Tensor boolValue = new Tensor(new byte[]{1, 1}, new int[]{2}, null, "boolValue", DataType.BOOL);
        Tensor floatValue = new Tensor(new float[]{3f, 4f}, new int[]{2}, null, "floatValue", DataType.FLOAT32);
        Tensor out = TensorPrimitiveBuilder.ternary(mask, boolValue, floatValue, new int[]{2}, new where(), "invalidWhere", DataType.FLOAT32);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PartitionPlanningContext context = planningContext(out);
        assertNull(new MetalRegionLegalityAdapter().tryCreateStructuralCandidate(
                operationNodeIds(context),
                context,
                Set.of(PartitionValueRef.ofNode(nodeId(context, Operation.OpType.WHERE)))
        ));
    }

    @Test
    void admitsDirectUnmaskedFloat32SdpaAfterNativeScaleParityVerification() {
        Tensor q = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "q", DataType.FLOAT32);
        Tensor k = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "k", DataType.FLOAT32);
        Tensor v = new Tensor(new float[]{10f, 1f, 1f, 10f}, new int[]{1, 2, 2}, null, "v", DataType.FLOAT32);
        Tensor out = q.scaledDotProductAttention(k, v, AttentionOptions.defaults().withScale(0.5));
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PartitionPlanningContext context = planningContext(out);
        int sdpaNodeId = nodeId(context, Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION);
        assertEquals("", MetalPartitionSupport.plannerUnsupportedReason(context.compiledNode(sdpaNodeId), context));
        assertNotNull(new MetalRegionLegalityAdapter().tryCreateStructuralCandidate(
                Set.of(sdpaNodeId),
                context,
                Set.of(PartitionValueRef.ofNode(sdpaNodeId))
        ));
    }

    @Test
    void directSdpaLowererEncodesScaleForFutureNativeEnablement() {
        Tensor q = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "q", DataType.FLOAT32);
        Tensor k = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "k", DataType.FLOAT32);
        Tensor v = new Tensor(new float[]{10f, 1f, 1f, 10f}, new int[]{1, 2, 2}, null, "v", DataType.FLOAT32);
        Tensor out = q.scaledDotProductAttention(k, v, AttentionOptions.defaults().withScale(0.5));
        PartitionPlanningContext context = planningContext(out);
        int sdpaNodeId = nodeId(context, Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION);
        CompiledNode sdpa = context.compiledNode(sdpaNodeId);

        var lowered = new AcceleratorSubgraphLowerer().tryLower(
                new AcceleratorSubgraphSpec(
                        sdpaNodeId,
                        List.of(sdpaNodeId),
                        List.of(new backend.accelerator.dag.AcceleratorSubgraphOp(sdpaNodeId, Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION)),
                        sdpa.inputIds(),
                        List.of(sdpaNodeId)
                ),
                context
        );

        assertNotNull(lowered);
        assertEquals(backend.accelerator.dag.AcceleratorDagNodeType.SDPA, lowered.dagSpec().nodes().getFirst().type());
        assertEquals(0.5f, Float.intBitsToFloat(lowered.dagSpec().nodes().getFirst().scalarValueBits()), 0.0f);
    }

    @Test
    void supportsDirectExternalBoolMaskedSdpaForMetal() {
        Tensor q = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "q", DataType.FLOAT32);
        Tensor k = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "k", DataType.FLOAT32);
        Tensor v = new Tensor(new float[]{10f, 1f, 1f, 10f}, new int[]{1, 2, 2}, null, "v", DataType.FLOAT32);
        Tensor mask = new Tensor(new byte[]{1, 0, 1, 1}, new int[]{1, 2, 2}, null, "mask", DataType.BOOL);
        Tensor out = q.scaledDotProductAttention(k, v, mask, AttentionOptions.defaults().withScale(0.5));
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PartitionPlanningContext context = planningContext(out);
        int sdpaNodeId = nodeId(context, Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION);
        assertEquals(
                "",
                MetalPartitionSupport.plannerUnsupportedReason(context.compiledNode(sdpaNodeId), context)
        );
        PartitionCandidate candidate = new MetalRegionLegalityAdapter().tryCreateStructuralCandidate(
                Set.of(sdpaNodeId),
                context,
                Set.of(PartitionValueRef.ofNode(sdpaNodeId))
        );
        assertNotNull(candidate);

        CompiledNode sdpa = context.compiledNode(sdpaNodeId);
        var lowered = new AcceleratorSubgraphLowerer().tryLower(
                new AcceleratorSubgraphSpec(
                        sdpaNodeId,
                        List.of(sdpaNodeId),
                        List.of(new backend.accelerator.dag.AcceleratorSubgraphOp(sdpaNodeId, Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION)),
                        sdpa.inputIds(),
                        List.of(sdpaNodeId)
                ),
                context
        );
        assertNotNull(lowered);
        assertEquals(backend.accelerator.dag.AcceleratorDagNodeType.SDPA, lowered.dagSpec().nodes().getFirst().type());
        assertEquals(backend.accelerator.dag.AcceleratorDagValueRefKind.EXTERNAL_INPUT, lowered.dagSpec().nodes().getFirst().input3().kind());
    }

    @Test
    void supportsDirectCausalSdpaViaEffectiveBoolMask() {
        Tensor q = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "q", DataType.FLOAT32);
        Tensor k = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "k", DataType.FLOAT32);
        Tensor v = new Tensor(new float[]{10f, 1f, 1f, 10f}, new int[]{1, 2, 2}, null, "v", DataType.FLOAT32);
        Tensor out = q.scaledDotProductAttention(k, v, AttentionOptions.causalDefaults().withScale(0.5));
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PartitionPlanningContext context = planningContext(out);
        int sdpaNodeId = nodeId(context, Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION);
        assertEquals(
                "",
                MetalPartitionSupport.plannerUnsupportedReason(context.compiledNode(sdpaNodeId), context)
        );
    }

    @Test
    void supportsDirectExternalAndCausalSdpaViaEffectiveBoolMask() {
        Tensor q = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "q", DataType.FLOAT32);
        Tensor k = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "k", DataType.FLOAT32);
        Tensor v = new Tensor(new float[]{10f, 1f, 1f, 10f}, new int[]{1, 2, 2}, null, "v", DataType.FLOAT32);
        Tensor mask = new Tensor(new byte[]{1, 0, 1, 1}, new int[]{1, 2, 2}, null, "mask", DataType.BOOL);
        Tensor out = q.scaledDotProductAttention(k, v, mask, AttentionOptions.causalDefaults().withScale(0.5));
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PartitionPlanningContext context = planningContext(out);
        int sdpaNodeId = nodeId(context, Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION);
        assertEquals(
                "",
                MetalPartitionSupport.plannerUnsupportedReason(context.compiledNode(sdpaNodeId), context)
        );
    }

    @Test
    void rejectsDirectBroadcastMaskSdpaWithMaskLayoutReason() {
        Tensor q = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "q", DataType.FLOAT32);
        Tensor k = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "k", DataType.FLOAT32);
        Tensor v = new Tensor(new float[]{10f, 1f, 1f, 10f}, new int[]{1, 2, 2}, null, "v", DataType.FLOAT32);
        Tensor mask = new Tensor(new byte[]{1, 0}, new int[]{1, 1, 2}, null, "mask", DataType.BOOL);
        Tensor out = q.scaledDotProductAttention(k, v, mask, AttentionOptions.defaults().withScale(0.5));
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PartitionPlanningContext context = planningContext(out);
        int sdpaNodeId = nodeId(context, Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION);
        assertEquals(
                "UNSUPPORTED_LAYOUT: GPU_METAL SDPA mask input requires dense BOOL layout",
                MetalPartitionSupport.plannerUnsupportedReason(context.compiledNode(sdpaNodeId), context)
        );
    }

    @Test
    void rejectsDirectFloat64SdpaForMetal() {
        Tensor q = new Tensor(new double[]{1d, 0d, 0d, 1d}, new int[]{1, 2, 2}, null, "q64", DataType.FLOAT64);
        Tensor k = new Tensor(new double[]{1d, 0d, 0d, 1d}, new int[]{1, 2, 2}, null, "k64", DataType.FLOAT64);
        Tensor v = new Tensor(new double[]{10d, 1d, 1d, 10d}, new int[]{1, 2, 2}, null, "v64", DataType.FLOAT64);
        Tensor out = q.scaledDotProductAttention(k, v, AttentionOptions.defaults().withScale(0.5));
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PartitionPlanningContext context = planningContext(out);
        int sdpaNodeId = nodeId(context, Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION);
        assertEquals(
                backend.metal.MetalMpsCapabilities.unsupportedDTypeMessage(DataType.FLOAT64),
                MetalPartitionSupport.plannerUnsupportedReason(context.compiledNode(sdpaNodeId), context)
        );
        assertNull(new MetalRegionLegalityAdapter().tryCreateStructuralCandidate(
                Set.of(sdpaNodeId),
                context,
                Set.of(PartitionValueRef.ofNode(sdpaNodeId))
        ));
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

    private static PartitionPlanningContext planningContext(Tensor out) {
        List<CompiledNode> compiledNodes = CompiledNode.snapshot(out.topologicalSort());
        return new PartitionPlanningContext(
                RuntimeConfig.inferenceDefaults(),
                false,
                compiledNodes,
                consumers(compiledNodes)
        );
    }

    private static Set<Integer> operationNodeIds(PartitionPlanningContext context) {
        return context.compiledNodes().stream()
                .filter(node -> node.operation() != null)
                .map(CompiledNode::id)
                .collect(java.util.stream.Collectors.toSet());
    }

    private static MetalPartitionPlan planFor(Tensor out, Operation.OpType opType) {
        PartitionPlanningContext context = planningContext(out);
        int nodeId = nodeId(context, opType);
        MetalRegionLegalityAdapter adapter = new MetalRegionLegalityAdapter();
        PartitionCandidate candidate = adapter.tryCreateStructuralCandidate(
                Set.of(nodeId),
                context,
                Set.of(PartitionValueRef.ofNode(nodeId))
        );
        assertNotNull(candidate);
        MetalPartitionPlan plan = (MetalPartitionPlan) adapter.tryCreatePlan(candidate, context);
        assertNotNull(plan);
        return plan;
    }

    private static int nodeId(PartitionPlanningContext context, Operation.OpType opType) {
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
