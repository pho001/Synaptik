package backend.metal.lowering;

import backend.ComputeBackend;
import backend.accelerator.dag.AcceleratorSubgraphSpec;
import backend.accelerator.lowering.AcceleratorSubgraphLowerer;
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

import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;

class MetalRegionLowererTest {
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
    void rejectsBfloat16MetalCandidateBeforeLowering() {
        Tensor a = new Tensor(new double[]{1d, 2d, 3d, 4d, 5d, 6d}, new int[]{2, 3}, null, "abf16", DataType.BFLOAT16);
        Tensor b = new Tensor(new double[]{1d, 2d, 3d, 4d, 5d, 6d}, new int[]{3, 2}, null, "bbf16", DataType.BFLOAT16);
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
    void rejectsDirectFloat32SdpaUntilNativeScaleContractMatchesCpu() {
        Tensor q = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "q", DataType.FLOAT32);
        Tensor k = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "k", DataType.FLOAT32);
        Tensor v = new Tensor(new float[]{10f, 1f, 1f, 10f}, new int[]{1, 2, 2}, null, "v", DataType.FLOAT32);
        Tensor out = q.scaledDotProductAttention(k, v, AttentionOptions.defaults().withScale(0.5));
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PartitionPlanningContext context = planningContext(out);
        int sdpaNodeId = nodeId(context, Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION);
        assertNull(new MetalRegionLegalityAdapter().tryCreateStructuralCandidate(
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
    void rejectsDirectMaskedSdpaForMetalUntilNativeMaskContractExists() {
        Tensor q = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "q", DataType.FLOAT32);
        Tensor k = new Tensor(new float[]{1f, 0f, 0f, 1f}, new int[]{1, 2, 2}, null, "k", DataType.FLOAT32);
        Tensor v = new Tensor(new float[]{10f, 1f, 1f, 10f}, new int[]{1, 2, 2}, null, "v", DataType.FLOAT32);
        Tensor mask = new Tensor(new byte[]{1, 0, 1, 1}, new int[]{1, 2, 2}, null, "mask", DataType.BOOL);
        Tensor out = q.scaledDotProductAttention(k, v, mask, AttentionOptions.defaults().withScale(0.5));
        TensorInternalAccess.setBackend(out, ComputeBackend.GPU_METAL);

        PartitionPlanningContext context = planningContext(out);
        int sdpaNodeId = nodeId(context, Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION);
        assertNull(new MetalRegionLegalityAdapter().tryCreateStructuralCandidate(
                Set.of(sdpaNodeId),
                context,
                Set.of(PartitionValueRef.ofNode(sdpaNodeId))
        ));
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
        assertNull(new MetalRegionLegalityAdapter().tryCreateStructuralCandidate(
                Set.of(sdpaNodeId),
                context,
                Set.of(PartitionValueRef.ofNode(sdpaNodeId))
        ));
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

    private static int nodeId(PartitionPlanningContext context, Operation.OpType opType) {
        return context.compiledNodes().stream()
                .filter(node -> node.operation() != null && node.operation().opType() == opType)
                .map(CompiledNode::id)
                .findFirst()
                .orElseThrow();
    }
}
