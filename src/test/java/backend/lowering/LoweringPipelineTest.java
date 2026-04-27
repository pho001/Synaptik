package backend.lowering;

import backend.ComputeBackend;
import backend.runtime.ExecutionMode;
import config.optimizer.FuseConfig;
import graph.CompiledNode;
import graph.execution.trace.PartitionDecisionTrace;
import graph.optimizer.memory.MemoryPlan;
import graph.optimizer.memory.MemoryPlanner;
import graph.optimizer.memory.MemoryPlannerPolicy;
import graph.optimizer.partition.Partition;
import graph.optimizer.partition.PartitionBoundaryReason;
import graph.optimizer.partition.PartitionEdge;
import graph.optimizer.partition.PartitionPlannerStrategy;
import graph.optimizer.partition.PartitionPlan;
import graph.optimizer.partition.PartitionTarget;
import graph.optimizer.partition.PartitionValue;
import graph.optimizer.partition.PartitionValueRef;
import graph.optimizer.region.DefaultRegionOptimizer;
import graph.optimizer.region.OptimizedRegion;
import graph.optimizer.region.RegionOptimizationContext;
import graph.optimizer.state.OptimizerState;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;

class LoweringPipelineTest {
    @Test
    void loweringPipelineRequiresMemoryPlan() {
        Tensor a = Tensor.scalar(1.0);
        OptimizerState state = OptimizerState.ofGraph(List.of(a), a);
        LoweringPipeline pipeline = new LoweringPipeline(List.of(request -> null));

        assertThrows(IllegalStateException.class, () ->
                pipeline.lower(state, BackendCapabilities.none(), new LoweringContext(null, List.of()))
        );
    }

    @Test
    void loweringPipelineBuildsLoweringStateFromOptimizedRegions() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{4}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{5f, 6f, 7f, 8f}, new int[]{4}, null, "b", DataType.FLOAT32);
        Tensor add = a.add(b);
        Tensor out = add.relu();

        List<Tensor> graph = out.topologicalSort();
        List<CompiledNode> compiledNodes = CompiledNode.snapshot(graph);
        Partition partition = partition(
                "cpu-partition",
                PartitionTarget.CPU,
                List.of(2, 3),
                List.of(0, 1),
                List.of(PartitionValueRef.ofNode(3)),
                List.of(PartitionValueRef.ofNode(3))
        );
        OptimizedRegion region = new DefaultRegionOptimizer().optimize(
                partition,
                new RegionOptimizationContext(compiledNodes, FuseConfig.inferenceDefaults())
        );
        MemoryPlan memoryPlan = MemoryPlanner.plan(graph, MemoryPlannerPolicy.defaults());
        OptimizerState optimized = OptimizerState.ofGraph(graph, out)
                .withExecutionMetadata(ExecutionMode.FORWARD, false, graph.indexOf(out))
                .withPartitions(List.of(partition))
                .withOptimizedRegions(List.of(region))
                .withMemoryPlan(memoryPlan);

        RegionLowerer lowerer = request -> new LoweringResult(
                new LoweredRegion(
                        request.region().regionId(),
                        request.region().target(),
                        request.region().executionUnits().stream()
                                .map(unit -> new LoweredExecutionUnit(
                                        unit.unitId(),
                                        LoweringFamily.DIRECT_KERNEL,
                                        unit.orderedNodeIds()
                                ))
                                .toList()
                ),
                List.of(new BackendWorkspaceRequirement("unit", "scratch", 16))
        );

        LoweringPipeline pipeline = new LoweringPipeline(List.of(lowerer));
        LoweringState lowered = pipeline.lower(
                optimized,
                new BackendCapabilities(Set.of(ComputeBackend.CPU)),
                new LoweringContext(null, compiledNodes)
        );

        assertEquals(optimized, lowered.optimized());
        assertEquals(1, lowered.lowered().loweredRegions().size());
        assertEquals(1, lowered.lowered().workspaceRequirements().size());
        assertFalse(lowered.trace().events().isEmpty());
    }

    @Test
    void loweringPipelineRespectsExplicitContextPartitionPlans() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{4}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{5f, 6f, 7f, 8f}, new int[]{4}, null, "b", DataType.FLOAT32);
        Tensor out = a.add(b).relu();

        List<Tensor> graph = out.topologicalSort();
        List<CompiledNode> compiledNodes = CompiledNode.snapshot(graph);
        Partition partition = partition(
                "metal-partition",
                PartitionTarget.GPU_METAL,
                List.of(2, 3),
                List.of(0, 1),
                List.of(PartitionValueRef.ofNode(3)),
                List.of(PartitionValueRef.ofNode(3))
        );
        OptimizedRegion region = new DefaultRegionOptimizer().optimize(
                partition,
                new RegionOptimizationContext(compiledNodes, FuseConfig.inferenceDefaults())
        );
        MemoryPlan memoryPlan = MemoryPlanner.plan(graph, MemoryPlannerPolicy.defaults());
        OptimizerState optimized = OptimizerState.ofGraph(graph, out)
                .withExecutionMetadata(ExecutionMode.FORWARD, false, graph.indexOf(out))
                .withPartitions(List.of(partition))
                .withOptimizedRegions(List.of(region))
                .withMemoryPlan(memoryPlan);

        PartitionPlan selectedPlan = new PartitionPlan() {
            @Override
            public ComputeBackend backend() {
                return ComputeBackend.GPU_METAL;
            }

            @Override
            public int anchorNodeId() {
                return 3;
            }

            @Override
            public List<Integer> nodeIds() {
                return List.of(2, 3);
            }

            @Override
            public List<Integer> externalInputNodeIds() {
                return List.of(0, 1);
            }

            @Override
            public List<Integer> producedOutputNodeIds() {
                return List.of(3);
            }

            @Override
            public long estimatedWork() {
                return 8L;
            }
        };

        RegionLowerer lowerer = request -> request.context().partitionPlanFor("metal-partition") == null
                ? null
                : new LoweringResult(
                        new LoweredRegion(request.region().regionId(), request.region().target(), List.of(
                                new LoweredExecutionUnit("gpu-unit", LoweringFamily.METAL_GRAPH_REGION, List.of(2, 3))
                        )),
                        List.of()
                );

        LoweringPipeline pipeline = new LoweringPipeline(List.of(lowerer));
        LoweringState lowered = pipeline.lower(
                optimized,
                new BackendCapabilities(Set.of(ComputeBackend.GPU_METAL)),
                new LoweringContext(null, compiledNodes, java.util.Map.of("metal-partition", selectedPlan))
        );

        assertEquals(1, lowered.lowered().loweredRegions().size());
        assertEquals(LoweringFamily.METAL_GRAPH_REGION, lowered.lowered().loweredRegions().getFirst().units().getFirst().loweringFamily());
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
                orderedNodeIds.getFirst(),
                requiredMaterialized,
                List.of(),
                List.of(PartitionBoundaryReason.NONE),
                orderedNodeIds.size(),
                new graph.optimizer.partition.cost.AcceleratorPartitionScoreModel.CandidateMetrics(orderedNodeIds.size(), internalEdges.size(), externalInputNodeIds.size(), 0, Math.max(0, orderedNodeIds.size() - 1)),
                PartitionPlannerStrategy.GREEDY_MAX_REGION,
                new PartitionDecisionTrace(
                        PartitionPlannerStrategy.GREEDY_MAX_REGION,
                        target,
                        orderedNodeIds.getFirst(),
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
}
