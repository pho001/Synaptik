package backend.apple.lowering;

import backend.ComputeBackend;
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
import graph.optimizer.partition.PartitionTarget;
import graph.optimizer.partition.PartitionValue;
import graph.optimizer.partition.PartitionValueRef;
import graph.optimizer.region.DefaultRegionOptimizer;
import graph.optimizer.region.ExecutionUnitKind;
import graph.optimizer.region.OptimizedRegion;
import graph.optimizer.region.RegionOptimizationContext;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import tensor.TensorInternalAccess;

import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;

class AppleRegionLowererTest {
    @Test
    void lowersGpuMetalRegionToAppleGraphRegion() {
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
        AppleGpuRegionLegalityAdapter adapter = new AppleGpuRegionLegalityAdapter();
        PartitionCandidate candidate = adapter.tryCreateStructuralCandidate(
                Set.of(2, 3),
                planningContext,
                Set.of(PartitionValueRef.ofNode(3))
        );
        var attachedPlan = adapter.tryCreatePlan(candidate, planningContext);
        Partition partition = new Partition(
                "apple-partition",
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

        AppleRegionLowerer lowerer = new AppleRegionLowerer();
        LoweringResult result = lowerer.lower(new LoweringRequest(
                region,
                MemoryPlanner.plan(graph),
                new BackendCapabilities(Set.of(ComputeBackend.GPU_METAL)),
                new LoweringContext(RuntimeConfig.inferenceDefaults(), compiledNodes, java.util.Map.of(partition.partitionId(), attachedPlan))
        ));

        assertNotNull(result);
        assertNotNull(result.loweredRegion());
        assertEquals(backend.lowering.LoweringFamily.APPLE_GRAPH_REGION, result.loweredRegion().units().getFirst().loweringFamily());
    }

    @Test
    void lowersPureElementwiseGpuMetalRegionToAppleFusedElementwiseGraph() {
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
        AppleGpuRegionLegalityAdapter adapter = new AppleGpuRegionLegalityAdapter();
        PartitionCandidate candidate = adapter.tryCreateStructuralCandidate(
                Set.of(2, 3, 4),
                planningContext,
                Set.of(PartitionValueRef.ofNode(4))
        );
        assertNotNull(candidate);
        var attachedPlan = adapter.tryCreatePlan(candidate, planningContext);
        assertNotNull(attachedPlan);

        Partition partition = new Partition(
                "apple-elementwise-partition",
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

        AppleRegionLowerer lowerer = new AppleRegionLowerer();
        LoweringResult result = lowerer.lower(new LoweringRequest(
                region,
                MemoryPlanner.plan(graph),
                new BackendCapabilities(Set.of(ComputeBackend.GPU_METAL)),
                new LoweringContext(RuntimeConfig.inferenceDefaults(), compiledNodes, java.util.Map.of(partition.partitionId(), attachedPlan))
        ));

        assertNotNull(result);
        assertNotNull(result.loweredRegion());
        assertEquals(backend.lowering.LoweringFamily.APPLE_FUSED_ELEMENTWISE_GRAPH, result.loweredRegion().units().getFirst().loweringFamily());
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
}
