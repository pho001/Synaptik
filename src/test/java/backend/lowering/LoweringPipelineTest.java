package backend.lowering;

import graph.compile.descriptor.CompiledTensorDescriptorBuilder;
import backend.contract.ComputeBackend;
import config.optimizer.FuseConfig;
import graph.compile.CompiledNodeSnapshotter;
import graph.model.CompiledNode;
import graph.execution.trace.PartitionDecisionTrace;
import graph.compile.planning.memory.MemoryPlan;
import graph.compile.planning.memory.MemoryPlanner;
import graph.compile.planning.memory.MemoryPlannerPolicy;
import graph.compile.planning.partition.Partition;
import graph.compile.planning.partition.PartitionBoundaryReason;
import graph.compile.planning.partition.PartitionEdge;
import graph.compile.planning.partition.PartitionPlannerStrategy;
import graph.compile.planning.partition.PartitionPlan;
import graph.compile.planning.partition.PartitionTarget;
import graph.compile.planning.partition.PartitionValue;
import graph.compile.planning.value.GraphValueRef;
import graph.compile.planning.region.DefaultRegionOptimizer;
import graph.compile.planning.region.OptimizedRegion;
import graph.compile.planning.region.RegionOptimizationContext;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import graph.compile.intent.BackendIntentPlan;

import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;

class LoweringPipelineTest {
    @Test
    void loweringInputRequiresMemoryPlan() {
        assertThrows(NullPointerException.class, () ->
                new LoweringInput(List.of(), null, java.util.Map.of())
        );
    }

    @Test
    void loweringPipelineBuildsLoweringStateFromOptimizedRegions() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{4}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{5f, 6f, 7f, 8f}, new int[]{4}, null, "b", DataType.FLOAT32);
        Tensor add = a.add(b);
        Tensor out = add.relu();

        List<Tensor> graph = out.topologicalSort();
        List<CompiledNode> compiledNodes = CompiledNodeSnapshotter.snapshot(graph, BackendIntentPlan.empty());
        Partition partition = partition(
                "cpu-partition",
                PartitionTarget.CPU,
                List.of(2, 3),
                List.of(0, 1),
                List.of(GraphValueRef.node(3)),
                List.of(GraphValueRef.node(3))
        );
        OptimizedRegion region = new DefaultRegionOptimizer().optimize(
                partition,
                new RegionOptimizationContext(compiledNodes, FuseConfig.inferenceDefaults())
        );
        MemoryPlan memoryPlan = MemoryPlanner.plan(graph, MemoryPlannerPolicy.defaults());
        LoweringInput input = new LoweringInput(List.of(region), memoryPlan, java.util.Map.of());

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
                input,
                new BackendCapabilities(Set.of(ComputeBackend.CPU)),
                new LoweringContext(null, compiledNodes, CompiledTensorDescriptorBuilder.build(compiledNodes), java.util.Map.of())
        );

        assertEquals(input, lowered.input());
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
        List<CompiledNode> compiledNodes = CompiledNodeSnapshotter.snapshot(graph, BackendIntentPlan.empty());
        Partition partition = partition(
                "metal-partition",
                PartitionTarget.GPU_METAL,
                List.of(2, 3),
                List.of(0, 1),
                List.of(GraphValueRef.node(3)),
                List.of(GraphValueRef.node(3))
        );
        OptimizedRegion region = new DefaultRegionOptimizer().optimize(
                partition,
                new RegionOptimizationContext(compiledNodes, FuseConfig.inferenceDefaults())
        );
        MemoryPlan memoryPlan = MemoryPlanner.plan(graph, MemoryPlannerPolicy.defaults());
        LoweringInput input = new LoweringInput(List.of(region), memoryPlan, java.util.Map.of());

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
                input,
                new BackendCapabilities(Set.of(ComputeBackend.GPU_METAL)),
                new LoweringContext(null, compiledNodes, CompiledTensorDescriptorBuilder.build(compiledNodes), java.util.Map.of("metal-partition", selectedPlan))
        );

        assertEquals(1, lowered.lowered().loweredRegions().size());
        assertEquals(LoweringFamily.METAL_GRAPH_REGION, lowered.lowered().loweredRegions().getFirst().units().getFirst().loweringFamily());
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
                orderedNodeIds.getFirst(),
                requiredMaterialized,
                List.of(),
                List.of(PartitionBoundaryReason.NONE),
                orderedNodeIds.size(),
                new graph.compile.planning.partition.cost.AcceleratorPartitionScoreModel.CandidateMetrics(orderedNodeIds.size(), internalEdges.size(), externalInputNodeIds.size(), 0, Math.max(0, orderedNodeIds.size() - 1)),
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
