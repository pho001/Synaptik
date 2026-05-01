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
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;
import tensor.TensorPrimitiveBuilder;

import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
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
    void cudaUnsupportedReductionUsesSharedUnsupportedReason() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "cudaReductionInput", DataType.FLOAT32);
        Tensor out = input.sum(1);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_CUDA);

        PartitionPlanningContext context = planningContext(out);
        String reason = CudaGpuRegionLegalityAdapter.plannerUnsupportedReason(context.compiledNode(nodeId(context, operations.Operation.OpType.SUM)), context);

        assertTrue(reason.contains("UNSUPPORTED_OPERATION"));
        assertTrue(reason.contains("operation SUM is not supported by GPU_CUDA lowering"));
    }

    @Test
    void cudaUnsupportedNormalizationUsesSharedUnsupportedReason() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "cudaNormInput", DataType.FLOAT32);
        Tensor gamma = new Tensor(new float[]{1f, 1f}, new int[]{2}, null, "cudaNormGamma", DataType.FLOAT32);
        Tensor beta = new Tensor(new float[]{0f, 0f}, new int[]{2}, null, "cudaNormBeta", DataType.FLOAT32);
        Tensor out = input.layerNorm(gamma, beta, 1.0e-5);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_CUDA);

        PartitionPlanningContext context = planningContext(out);
        String reason = CudaGpuRegionLegalityAdapter.plannerUnsupportedReason(context.compiledNode(nodeId(context, operations.Operation.OpType.LAYER_NORM)), context);

        assertTrue(reason.contains("DEFERRED_FUSED_REGION"));
        assertTrue(reason.contains("operation LAYER_NORM is not supported by GPU_CUDA lowering"));
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

        assertTrue(reason.contains("UNSUPPORTED_DTYPE"));
        assertTrue(reason.contains("operation CROSS_ENTROPY_LOSS_INDICES is not supported by GPU_CUDA lowering"));
    }

    @Test
    void cudaRejectsDirectNonDenseComputeUntilLayoutIsMaterialized() {
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
    void rejectsSumReductionWithStableCoverageReason() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "cudaStableReductionInput", DataType.FLOAT32);
        Tensor out = input.sum(1);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_CUDA);

        PartitionPlanningContext context = planningContext(out);
        String reason = CudaGpuRegionLegalityAdapter.plannerUnsupportedReason(context.compiledNode(nodeId(context, operations.Operation.OpType.SUM)), context);

        assertTrue(reason.contains("UNSUPPORTED_OPERATION"));
    }

    @Test
    void rejectsLayerNormWithStableCoverageReason() {
        Tensor input = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{2, 2}, null, "cudaStableNormInput", DataType.FLOAT32);
        Tensor gamma = new Tensor(new float[]{1f, 1f}, new int[]{2}, null, "cudaStableNormGamma", DataType.FLOAT32);
        Tensor beta = new Tensor(new float[]{0f, 0f}, new int[]{2}, null, "cudaStableNormBeta", DataType.FLOAT32);
        Tensor out = input.layerNorm(gamma, beta, 1.0e-5);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_CUDA);

        PartitionPlanningContext context = planningContext(out);
        String reason = CudaGpuRegionLegalityAdapter.plannerUnsupportedReason(context.compiledNode(nodeId(context, operations.Operation.OpType.LAYER_NORM)), context);

        assertTrue(reason.contains("DEFERRED_FUSED_REGION"));
    }

    @Test
    void rejectsCrossEntropyLossWithStableCoverageReason() {
        Tensor logits = new Tensor(new float[]{1f, 2f, 3f, 1f, 0f, -1f}, new int[]{2, 3}, null, "cudaStableLossLogits", DataType.FLOAT32);
        Tensor targetIndices = new Tensor(new int[]{2, 0}, new int[]{2}, null, "cudaStableLossTargets", DataType.INT32);
        Tensor out = logits.crossEntropyLossFromIndices(targetIndices, 1);
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_CUDA);

        PartitionPlanningContext context = planningContext(out);
        String reason = CudaGpuRegionLegalityAdapter.plannerUnsupportedReason(context.compiledNode(nodeId(context, operations.Operation.OpType.CROSS_ENTROPY_LOSS_INDICES)), context);

        assertTrue(reason.contains("UNSUPPORTED_DTYPE"));
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
