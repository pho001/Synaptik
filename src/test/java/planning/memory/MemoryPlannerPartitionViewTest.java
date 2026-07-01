package planning.memory;

import config.optimizer.FuseConfig;
import config.compile.CompileConfig;
import graph.CompiledGraph;
import graph.compile.CompiledNodeSnapshotter;
import graph.model.CompiledNode;
import trace.compile.PartitionDecisionTrace;
import planning.partition.Partition;
import planning.partition.ExecutablePartitionPlan;
import planning.partition.PlannedPartition;
import planning.partition.PartitionBoundaryReason;
import planning.partition.PartitionEdge;
import planning.partition.PartitionPlannerStrategy;
import planning.partition.PartitionTarget;
import planning.partition.PartitionValue;
import planning.value.GraphValueRef;
import planning.partition.execution.PartitionExecutionPlanner;
import planning.partition.execution.ExecutionUnit;
import planning.partition.execution.ExecutionUnitKind;
import planning.partition.execution.MaterializationDecision;
import planning.partition.execution.PartitionExecutionPlan;
import planning.partition.execution.PartitionExecutionTrace;
import planning.partition.execution.PartitionExecutionPlanningContext;
import planning.partition.execution.PartitionExecutionValue;
import planning.partition.execution.ValueTransportKind;
import planning.partition.execution.ValueTypeContract;
import planning.partition.specialization.PartitionSpecializationDecision;
import planning.partition.specialization.PartitionSpecializationKind;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import planning.intent.BackendIntentPlan;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class MemoryPlannerPartitionViewTest {
    @Test
    void memoryPlannerCapturesStructuralViewFromPartitionExecutionPlans() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{4}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{5f, 6f, 7f, 8f}, new int[]{4}, null, "b", DataType.FLOAT32);
        Tensor add = a.add(b);
        Tensor out = add.relu();

        List<Tensor> graph = out.topologicalSort();
        List<CompiledNode> compiledNodes = CompiledNodeSnapshotter.snapshot(graph, BackendIntentPlan.empty());
        Partition partition = partition(
                "cpu-cont",
                PartitionTarget.CPU,
                List.of(2, 3),
                List.of(0, 1),
                List.of(GraphValueRef.node(3)),
                List.of()
        );
        PartitionExecutionPlan executionPlan = new PartitionExecutionPlanner().planPartition(
                partition,
                new PartitionExecutionPlanningContext(compiledNodes, FuseConfig.inferenceDefaults())
        );

        MemoryPlan plan = planFor(graph, List.of(executable(partition, executionPlan)), graph.indexOf(out));

        assertNotNull(plan);
        assertEquals(1, plan.structuralView().plannedPartitionIds().size());
        assertEquals(1, plan.structuralView().continuationValues().size());
        GraphValueRef outputRef = executionPlan.executionValues().stream()
                .filter(value -> value.transportKind() == ValueTransportKind.CONTINUATION)
                .map(PartitionExecutionValue::ref)
                .findFirst()
                .orElseThrow();
        assertEquals(outputRef, plan.graphValueRefOfNodeId(graph.indexOf(out)));
        assertEquals(MaterializationDecision.CONTINUE, plan.materializationPlanOf(outputRef).decision());
        assertEquals(PartitionMemoryBindingKind.CONTINUATION, plan.partitionMemoryBindingOf(outputRef).kind());
        assertEquals(plan.partitionSlotIdOf(outputRef), plan.runtimeSlotIdOfNodeId(graph.indexOf(out)));
        assertNotNull(plan.partitionSlotIdOf(outputRef));
        assertEquals(4, plan.partitionSlotSize(plan.partitionSlotIdOf(outputRef)));
        assertTrue(plan.explain().contains("=== Structural Memory View ==="));
        assertTrue(plan.explain().contains("=== Partition Value Plan ==="));
    }

    @Test
    void memoryPlannerCapturesCrossPartitionExecutionValueFlow() {
        Tensor a = Tensor.scalar(1.0f);
        Tensor b = Tensor.scalar(2.0f);
        Tensor mid = a.add(b);
        Tensor out = mid.relu();
        List<Tensor> graph = out.topologicalSort();
        int aNodeId = graph.indexOf(a);
        int bNodeId = graph.indexOf(b);
        int midNodeId = graph.indexOf(mid);
        int outNodeId = graph.indexOf(out);
        GraphValueRef shared = GraphValueRef.node(midNodeId);
        GraphValueRef finalOut = GraphValueRef.node(outNodeId);

        Partition producerPartition = partition(
                "partition-a", PartitionTarget.CPU, List.of(midNodeId), List.of(aNodeId, bNodeId), List.of(shared), List.of());
        PartitionExecutionPlan producer = new PartitionExecutionPlan(
                List.of(new ExecutionUnit(
                        "unit-a",
                        ExecutionUnitKind.SINGLE_OP,
                        PartitionTarget.CPU,
                        List.of(GraphValueRef.node(aNodeId), GraphValueRef.node(bNodeId)),
                        List.of(shared),
                        List.of(),
                        List.of(shared),
                        List.of(midNodeId),
                        2L,
                        List.of(aNodeId, bNodeId),
                        PartitionExecutionTrace.empty()
                )),
                List.of(new PartitionExecutionValue(
                        shared,
                        shared,
                        midNodeId,
                        1,
                        ValueTransportKind.CONTINUATION,
                        ValueTypeContract.same(DataType.FLOAT32),
                        false
                )),
                List.of(),
                PartitionExecutionTrace.empty()
        );
        Partition consumerPartition = partition(
                "partition-b", PartitionTarget.CPU, List.of(outNodeId), List.of(midNodeId), List.of(finalOut), List.of(finalOut));
        PartitionExecutionPlan consumer = new PartitionExecutionPlan(
                List.of(new ExecutionUnit(
                        "unit-b",
                        ExecutionUnitKind.SINGLE_OP,
                        PartitionTarget.CPU,
                        List.of(shared),
                        List.of(finalOut),
                        List.of(finalOut),
                        List.of(),
                        List.of(outNodeId),
                        2L,
                        List.of(midNodeId),
                        PartitionExecutionTrace.empty()
                )),
                List.of(new PartitionExecutionValue(
                        finalOut,
                        finalOut,
                        outNodeId,
                        1,
                        ValueTransportKind.MATERIALIZED,
                        ValueTypeContract.same(DataType.FLOAT32),
                        true
                )),
                List.of(finalOut),
                PartitionExecutionTrace.empty()
        );

        MemoryPlan plan = planFor(graph, List.of(
                executable(producerPartition, producer),
                executable(consumerPartition, consumer)
        ), outNodeId);

        assertNotNull(plan);
        assertEquals(1, plan.structuralView().crossPartitionDependencyCount());
        var flow = plan.structuralView().flowOf(shared);
        assertNotNull(flow);
        assertEquals(MaterializationDecision.CONTINUE, flow.decision());
        assertEquals(List.of("partition-b"), flow.consumerPartitionIds());
        assertEquals(List.of("unit-b"), flow.consumerUnitIds());
        assertEquals(finalOut, plan.graphValueRefOfNodeId(finalOut.nodeId()));
        assertEquals(0, plan.partitionValueLifetimeOf(shared).birthStep());
        assertEquals(outNodeId, plan.partitionValueLifetimeOf(shared).lastUseStep());
        assertTrue(plan.partitionValueLifetimeOf(shared).isCrossPartition());
        assertEquals(PartitionMemoryBindingKind.CONTINUATION, plan.partitionMemoryBindingOf(shared).kind());
        assertEquals(plan.partitionSlotIdOf(finalOut), plan.runtimeSlotIdOfNodeId(finalOut.nodeId()));
        assertEquals(1, plan.partitionSlotSize(plan.partitionSlotIdOf(shared)));
        assertEquals(1, plan.handoffRequirements().size());
    }

    @Test
    void terminalGradientTargetsDoNotReusePartitionBindings() throws Exception {
        Tensor a = new Tensor(new double[]{1, 5, 3}, new int[]{3}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{2, 4, 3}, new int[]{3}, null, "b", DataType.FLOAT64);
        a.setRequiresGrad(true);
        b.setRequiresGrad(true);

        Tensor min = a.minimum(b);
        CompiledGraph compiled = CompiledGraph.compile(min, CompileConfig.training());

        MemoryPlan plan = compiled.program().memoryPlan();

        assertNotNull(plan);
        List<GraphValueRef> materializedValues = plan.structuralView().materializedValues();
        List<GraphValueRef> materializedGradientValues = compiled.program().compiledNodes().stream()
                .filter(CompiledNode::backwardNode)
                .map(node -> GraphValueRef.node(node.id()))
                .filter(materializedValues::contains)
                .toList();
        assertEquals(2, materializedGradientValues.size());
        assertTrue(materializedValues.stream()
                .allMatch(valueRef -> plan.partitionValueLifetimeOf(valueRef).lastUseStep() == compiled.program().compiledNodes().size()));
        long distinctSlotCount = materializedValues.stream()
                .map(plan::partitionSlotIdOf)
                .distinct()
                .count();
        assertEquals(materializedValues.size(), distinctSlotCount);
    }

    @Test
    void memoryPlannerKeepsStructuralMemoryPlanForBfloat16Graphs() {
        Tensor a = new Tensor(new short[]{1, 2, 3, 4}, new int[]{4}, null, "a", DataType.BFLOAT16);
        Tensor b = new Tensor(new short[]{5, 6, 7, 8}, new int[]{4}, null, "b", DataType.BFLOAT16);
        Tensor add = a.add(b);
        Tensor out = add.relu();

        List<Tensor> graph = out.topologicalSort();
        List<CompiledNode> compiledNodes = CompiledNodeSnapshotter.snapshot(graph, BackendIntentPlan.empty());
        Partition partition = partition(
                "bf16-partition",
                PartitionTarget.CPU,
                List.of(2, 3),
                List.of(0, 1),
                List.of(GraphValueRef.node(3)),
                List.of()
        );
        PartitionExecutionPlan executionPlan = new PartitionExecutionPlanner().planPartition(
                partition,
                new PartitionExecutionPlanningContext(compiledNodes, FuseConfig.inferenceDefaults())
        );

        MemoryPlan plan = planFor(graph, List.of(executable(partition, executionPlan)), graph.indexOf(out));

        assertNotNull(plan);
        assertEquals(1, plan.structuralView().plannedPartitionIds().size());
    }

    @Test
    void memoryPlannerVirtualizesAcceptedMseIntermediateValues() {
        Tensor pred = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{4}, null, "memoryMsePred", DataType.FLOAT32);
        Tensor target = new Tensor(new float[]{1.5f, 1f, 2.5f, 3f}, new int[]{4}, null, "memoryMseTarget", DataType.FLOAT32);
        Tensor diff = pred.sub(target);
        Tensor square = diff.mul(diff);
        Tensor loss = square.mean();

        List<Tensor> graph = loss.topologicalSort();
        int diffNodeId = graph.indexOf(diff);
        int squareNodeId = graph.indexOf(square);
        int lossNodeId = graph.indexOf(loss);
        List<CompiledNode> compiledNodes = CompiledNodeSnapshotter.snapshot(graph, BackendIntentPlan.empty());
        Partition partition = partition(
                "memory-cpu-mse",
                PartitionTarget.CPU,
                List.of(diffNodeId, squareNodeId, lossNodeId),
                List.of(graph.indexOf(pred), graph.indexOf(target)),
                List.of(GraphValueRef.node(lossNodeId)),
                List.of(GraphValueRef.node(lossNodeId))
        );
        PartitionExecutionPlan executionPlan = new PartitionExecutionPlanner((partitionTarget, candidate) ->
                partitionTarget == PartitionTarget.CPU && candidate.kind() == PartitionSpecializationKind.MSE_LOSS
                        ? PartitionSpecializationDecision.accept("test-cpu-mse-specialization")
                        : PartitionSpecializationDecision.reject("test-reject")
        ).planPartition(
                partition,
                new PartitionExecutionPlanningContext(compiledNodes, FuseConfig.inferenceDefaults())
        );

        MemoryPlan plan = planFor(graph, List.of(executable(partition, executionPlan)), lossNodeId);

        assertEquals(1, executionPlan.executionUnits().size());
        assertEquals(ExecutionUnitKind.SPECIALIZED_PRIMITIVE, executionPlan.executionUnits().getFirst().kind());
        assertEquals(MaterializationDecision.VIRTUALIZE,
                plan.materializationPlanOf(GraphValueRef.node(diffNodeId)).decision());
        assertEquals(MaterializationDecision.VIRTUALIZE,
                plan.materializationPlanOf(GraphValueRef.node(squareNodeId)).decision());
        assertFalse(plan.materializationPlanOf(GraphValueRef.node(diffNodeId)).allocatesStorage());
        assertFalse(plan.materializationPlanOf(GraphValueRef.node(squareNodeId)).allocatesStorage());
        assertEquals(PartitionMemoryBindingKind.NONE, plan.partitionMemoryBindingOf(GraphValueRef.node(diffNodeId)).kind());
        assertEquals(PartitionMemoryBindingKind.NONE, plan.partitionMemoryBindingOf(GraphValueRef.node(squareNodeId)).kind());
        assertEquals(MaterializationDecision.MATERIALIZE,
                plan.materializationPlanOf(GraphValueRef.node(lossNodeId)).decision());
    }

    @Test
    void virtualValuesRemainUnallocatedInPartitionMemoryPlan() {
        GraphValueRef virtualValue = GraphValueRef.node(31);
        GraphValueRef materializedValue = GraphValueRef.node(32);
        Tensor a = Tensor.scalar(1.0f);
        Tensor b = a.relu();

        Partition sourcePartition = partition(
                "partition-virtual", PartitionTarget.CPU, List.of(30, 31, 32), List.of(0), List.of(GraphValueRef.node(32)), List.of(GraphValueRef.node(32)));
        PartitionExecutionPlan executionPlan = new PartitionExecutionPlan(
                List.of(
                        new ExecutionUnit(
                                "unit-virtual",
                                ExecutionUnitKind.SINGLE_OP,
                                PartitionTarget.CPU,
                                List.of(GraphValueRef.node(0)),
                                List.of(virtualValue),
                                List.of(),
                                List.of(virtualValue),
                                List.of(30, 31),
                                2L,
                                List.of(0),
                                PartitionExecutionTrace.empty()
                        ),
                        new ExecutionUnit(
                                "unit-materialized",
                                ExecutionUnitKind.SINGLE_OP,
                                PartitionTarget.CPU,
                                List.of(virtualValue),
                                List.of(materializedValue),
                                List.of(materializedValue),
                                List.of(),
                                List.of(32),
                                1L,
                                List.of(),
                                PartitionExecutionTrace.empty()
                        )
                ),
                List.of(
                        new PartitionExecutionValue(
                                virtualValue,
                                GraphValueRef.node(31),
                                31,
                                1,
                                ValueTransportKind.VIRTUAL,
                                ValueTypeContract.same(DataType.FLOAT32),
                                false
                        ),
                        new PartitionExecutionValue(
                                materializedValue,
                                GraphValueRef.node(32),
                                32,
                                1,
                                ValueTransportKind.MATERIALIZED,
                                ValueTypeContract.same(DataType.FLOAT32),
                                true
                        )
                ),
                List.of(materializedValue),
                PartitionExecutionTrace.empty()
        );

        List<Tensor> graph = b.topologicalSort();
        MemoryPlan plan = planFor(graph, List.of(executable(sourcePartition, executionPlan)), graph.indexOf(b));

        assertNotNull(plan);
        assertEquals(MaterializationDecision.VIRTUALIZE, plan.materializationPlanOf(virtualValue).decision());
        assertFalse(plan.materializationPlanOf(virtualValue).allocatesStorage());
        assertEquals(PartitionMemoryBindingKind.NONE, plan.partitionMemoryBindingOf(virtualValue).kind());
        assertEquals(MaterializationDecision.MATERIALIZE, plan.materializationPlanOf(materializedValue).decision());
        assertEquals(materializedValue, plan.graphValueRefOfNodeId(materializedValue.nodeId()));
        assertTrue(plan.partitionMemoryBindingOf(materializedValue).hasBindingId());
        assertEquals(plan.partitionSlotIdOf(materializedValue), plan.runtimeSlotIdOfNodeId(materializedValue.nodeId()));
        assertEquals(1, plan.partitionSlotSize(plan.partitionSlotIdOf(materializedValue)));
    }

    private static MemoryPlan planFor(
            List<Tensor> graph,
            List<ExecutablePartitionPlan> partitions,
            int forwardBoundaryNodeId
    ) {
        return MemoryPlanner.plan(
                new MemoryPlanningInput(
                        CompiledNodeSnapshotter.snapshot(graph, BackendIntentPlan.empty()),
                        partitions,
                        false,
                        forwardBoundaryNodeId
                ),
                MemoryPlannerPolicy.defaults()
        );
    }

    private static ExecutablePartitionPlan executable(Partition partition, PartitionExecutionPlan executionPlan) {
        return new ExecutablePartitionPlan(new PlannedPartition(partition, null, java.util.Set.of()), executionPlan);
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
