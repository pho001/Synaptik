package backend.lowering;

import planning.descriptor.CompiledTensorDescriptorBuilder;
import backend.contract.ComputeBackend;
import config.optimizer.FuseConfig;
import graph.compile.CompiledNodeSnapshotter;
import graph.model.CompiledNode;
import trace.compile.PartitionDecisionTrace;
import planning.memory.MemoryPlan;
import planning.memory.MemoryPlanner;
import planning.memory.MemoryPlannerPolicy;
import planning.partition.Partition;
import planning.partition.ExecutablePartitionPlan;
import planning.partition.PlannedPartition;
import planning.partition.PartitionBoundaryReason;
import planning.partition.PartitionEdge;
import planning.partition.PartitionPlannerStrategy;
import planning.partition.PartitionPlan;
import planning.partition.PartitionTarget;
import planning.partition.PartitionValue;
import planning.value.GraphValueRef;
import planning.partition.execution.PartitionExecutionPlanner;
import planning.partition.execution.PartitionExecutionPlan;
import planning.partition.execution.PartitionExecutionPlanningContext;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import planning.intent.BackendIntentPlan;

import java.util.List;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;

class LoweringPipelineTest {
    @Test
    void loweringInputRequiresMemoryPlan() {
        assertThrows(NullPointerException.class, () ->
                new LoweringInput(List.of(), null)
        );
    }

    @Test
    void loweringPipelineBuildsLoweringStateFromPartitionExecutionPlans() {
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
        PartitionExecutionPlan executionPlan = new PartitionExecutionPlanner().planPartition(
                partition,
                new PartitionExecutionPlanningContext(compiledNodes, FuseConfig.inferenceDefaults())
        );
        MemoryPlan memoryPlan = MemoryPlanner.plan(graph, MemoryPlannerPolicy.defaults());
        ExecutablePartitionPlan executablePartition = new ExecutablePartitionPlan(
                new PlannedPartition(partition, null, Set.of(ComputeBackend.CPU)), executionPlan);
        LoweringInput input = new LoweringInput(List.of(executablePartition), memoryPlan);

        PartitionLowerer lowerer = request -> new LoweringResult(
                new LoweredPartition(
                        request.executablePartition(),
                        request.executablePartition().executionPlan().executionUnits().stream()
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
                new LoweringContext(null, compiledNodes, CompiledTensorDescriptorBuilder.build(compiledNodes))
        );

        assertEquals(input, lowered.input());
        assertEquals(1, lowered.lowered().loweredPartitions().size());
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
        PartitionExecutionPlan executionPlan = new PartitionExecutionPlanner().planPartition(
                partition,
                new PartitionExecutionPlanningContext(compiledNodes, FuseConfig.inferenceDefaults())
        );
        MemoryPlan memoryPlan = MemoryPlanner.plan(graph, MemoryPlannerPolicy.defaults());
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

        ExecutablePartitionPlan executablePartition = new ExecutablePartitionPlan(
                new PlannedPartition(partition, selectedPlan, Set.of(ComputeBackend.GPU_METAL)), executionPlan);
        LoweringInput input = new LoweringInput(List.of(executablePartition), memoryPlan);

        PartitionLowerer lowerer = request -> request.executablePartition().backendPlan() == null
                ? null
                : new LoweringResult(
                        new LoweredPartition(request.executablePartition(), List.of(
                                new LoweredExecutionUnit("gpu-unit", LoweringFamily.METAL_GRAPH_PARTITION, List.of(2, 3))
                        )),
                        List.of()
                );

        LoweringPipeline pipeline = new LoweringPipeline(List.of(lowerer));
        LoweringState lowered = pipeline.lower(
                input,
                new BackendCapabilities(Set.of(ComputeBackend.GPU_METAL)),
                new LoweringContext(null, compiledNodes, CompiledTensorDescriptorBuilder.build(compiledNodes))
        );

        assertEquals(1, lowered.lowered().loweredPartitions().size());
        assertEquals(LoweringFamily.METAL_GRAPH_PARTITION, lowered.lowered().loweredPartitions().getFirst().units().getFirst().loweringFamily());
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
                new planning.partition.cost.AcceleratorPartitionScoreModel.CandidateMetrics(orderedNodeIds.size(), internalEdges.size(), externalInputNodeIds.size(), 0, Math.max(0, orderedNodeIds.size() - 1)),
                PartitionPlannerStrategy.GREEDY_MAX_PARTITION,
                new PartitionDecisionTrace(
                        PartitionPlannerStrategy.GREEDY_MAX_PARTITION.name(),
                        target.name(),
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
