package graph.compile.planning.memory;

import backend.runtime.ExecutionMode;
import config.optimizer.FuseConfig;
import config.compile.CompileConfig;
import graph.CompiledGraph;
import graph.compile.CompiledNodeSnapshotter;
import graph.model.CompiledNode;
import graph.execution.trace.PartitionDecisionTrace;
import graph.compile.planning.partition.Partition;
import graph.compile.planning.partition.PartitionBoundaryReason;
import graph.compile.planning.partition.PartitionEdge;
import graph.compile.planning.partition.PartitionPlannerStrategy;
import graph.compile.planning.partition.PartitionTarget;
import graph.compile.planning.partition.PartitionValue;
import graph.compile.planning.value.GraphValueRef;
import graph.compile.planning.region.DefaultRegionOptimizer;
import graph.compile.planning.region.ExecutionUnit;
import graph.compile.planning.region.ExecutionUnitKind;
import graph.compile.planning.region.MaterializationDecision;
import graph.compile.planning.region.OptimizedRegion;
import graph.compile.planning.region.RegionOptimizationTrace;
import graph.compile.planning.region.RegionOptimizationContext;
import graph.compile.planning.region.RegionValue;
import graph.compile.planning.region.ValueTransportKind;
import graph.compile.planning.region.ValueTypeContract;
import graph.compile.planning.region.specialization.RegionSpecializationDecision;
import graph.compile.planning.region.specialization.RegionSpecializationKind;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import graph.compile.intent.BackendIntentPlan;

import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class MemoryPlannerRegionViewTest {
    @Test
    void memoryPlannerCapturesStructuralViewFromOptimizedRegions() {
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
        OptimizedRegion region = new DefaultRegionOptimizer().optimize(
                partition,
                new RegionOptimizationContext(compiledNodes, FuseConfig.inferenceDefaults())
        );

        MemoryPlan plan = planFor(graph, List.of(region), graph.indexOf(out));

        assertNotNull(plan);
        assertEquals(1, plan.structuralView().optimizedRegionIds().size());
        assertEquals(1, plan.structuralView().continuationValues().size());
        GraphValueRef outputRef = region.regionValues().stream()
                .filter(value -> value.transportKind() == ValueTransportKind.CONTINUATION)
                .map(RegionValue::ref)
                .findFirst()
                .orElseThrow();
        assertEquals(outputRef, plan.graphValueRefOfNodeId(graph.indexOf(out)));
        assertEquals(MaterializationDecision.CONTINUE, plan.materializationPlanOf(outputRef).decision());
        assertEquals(RegionMemoryBindingKind.CONTINUATION, plan.regionMemoryBindingOf(outputRef).kind());
        assertEquals(plan.regionSlotIdOf(outputRef), plan.runtimeSlotIdOfNodeId(graph.indexOf(out)));
        assertNotNull(plan.regionSlotIdOf(outputRef));
        assertEquals(4, plan.regionSlotSize(plan.regionSlotIdOf(outputRef)));
        assertTrue(plan.explain().contains("=== Structural Memory View ==="));
        assertTrue(plan.explain().contains("=== Region Value Plan ==="));
    }

    @Test
    void memoryPlannerCapturesCrossRegionValueFlow() {
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

        OptimizedRegion producer = new OptimizedRegion(
                "region-a",
                partition("region-a", PartitionTarget.CPU, List.of(midNodeId), List.of(aNodeId, bNodeId), List.of(shared), List.of()),
                PartitionTarget.CPU,
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
                        RegionOptimizationTrace.empty()
                )),
                List.of(new RegionValue(
                        shared,
                        shared,
                        midNodeId,
                        1,
                        ValueTransportKind.CONTINUATION,
                        ValueTypeContract.same(DataType.FLOAT32),
                        false
                )),
                List.of(),
                RegionOptimizationTrace.empty()
        );
        OptimizedRegion consumer = new OptimizedRegion(
                "region-b",
                partition("region-b", PartitionTarget.CPU, List.of(outNodeId), List.of(midNodeId), List.of(finalOut), List.of(finalOut)),
                PartitionTarget.CPU,
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
                        RegionOptimizationTrace.empty()
                )),
                List.of(new RegionValue(
                        finalOut,
                        finalOut,
                        outNodeId,
                        1,
                        ValueTransportKind.MATERIALIZED,
                        ValueTypeContract.same(DataType.FLOAT32),
                        true
                )),
                List.of(finalOut),
                RegionOptimizationTrace.empty()
        );

        MemoryPlan plan = planFor(graph, List.of(producer, consumer), outNodeId);

        assertNotNull(plan);
        assertEquals(1, plan.structuralView().crossRegionDependencyCount());
        var flow = plan.structuralView().flowOf(shared);
        assertNotNull(flow);
        assertEquals(MaterializationDecision.CONTINUE, flow.decision());
        assertEquals(List.of("region-b"), flow.consumerRegionIds());
        assertEquals(List.of("unit-b"), flow.consumerUnitIds());
        assertEquals(finalOut, plan.graphValueRefOfNodeId(finalOut.nodeId()));
        assertEquals(0, plan.regionValueLifetimeOf(shared).birthStep());
        assertEquals(outNodeId, plan.regionValueLifetimeOf(shared).lastUseStep());
        assertTrue(plan.regionValueLifetimeOf(shared).isCrossRegion());
        assertEquals(RegionMemoryBindingKind.CONTINUATION, plan.regionMemoryBindingOf(shared).kind());
        assertEquals(plan.regionSlotIdOf(finalOut), plan.runtimeSlotIdOfNodeId(finalOut.nodeId()));
        assertEquals(1, plan.regionSlotSize(plan.regionSlotIdOf(shared)));
        assertEquals(1, plan.handoffRequirements().size());
    }

    @Test
    void terminalGradientTargetsDoNotReuseRegionBindings() throws Exception {
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
                .allMatch(valueRef -> plan.regionValueLifetimeOf(valueRef).lastUseStep() == compiled.program().compiledNodes().size()));
        long distinctSlotCount = materializedValues.stream()
                .map(plan::regionSlotIdOf)
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
                "bf16-region",
                PartitionTarget.CPU,
                List.of(2, 3),
                List.of(0, 1),
                List.of(GraphValueRef.node(3)),
                List.of()
        );
        OptimizedRegion region = new DefaultRegionOptimizer().optimize(
                partition,
                new RegionOptimizationContext(compiledNodes, FuseConfig.inferenceDefaults())
        );

        MemoryPlan plan = planFor(graph, List.of(region), graph.indexOf(out));

        assertNotNull(plan);
        assertEquals(1, plan.structuralView().optimizedRegionIds().size());
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
        OptimizedRegion region = new DefaultRegionOptimizer((partitionTarget, candidate) ->
                partitionTarget == PartitionTarget.CPU && candidate.kind() == RegionSpecializationKind.MSE_LOSS
                        ? RegionSpecializationDecision.accept("test-cpu-mse-specialization")
                        : RegionSpecializationDecision.reject("test-reject")
        ).optimize(
                partition,
                new RegionOptimizationContext(compiledNodes, FuseConfig.inferenceDefaults())
        );

        MemoryPlan plan = planFor(graph, List.of(region), lossNodeId);

        assertEquals(1, region.executionUnits().size());
        assertEquals(ExecutionUnitKind.SPECIALIZED_PRIMITIVE, region.executionUnits().getFirst().kind());
        assertEquals(MaterializationDecision.VIRTUALIZE,
                plan.materializationPlanOf(GraphValueRef.node(diffNodeId)).decision());
        assertEquals(MaterializationDecision.VIRTUALIZE,
                plan.materializationPlanOf(GraphValueRef.node(squareNodeId)).decision());
        assertFalse(plan.materializationPlanOf(GraphValueRef.node(diffNodeId)).allocatesStorage());
        assertFalse(plan.materializationPlanOf(GraphValueRef.node(squareNodeId)).allocatesStorage());
        assertEquals(RegionMemoryBindingKind.NONE, plan.regionMemoryBindingOf(GraphValueRef.node(diffNodeId)).kind());
        assertEquals(RegionMemoryBindingKind.NONE, plan.regionMemoryBindingOf(GraphValueRef.node(squareNodeId)).kind());
        assertEquals(MaterializationDecision.MATERIALIZE,
                plan.materializationPlanOf(GraphValueRef.node(lossNodeId)).decision());
    }

    @Test
    void virtualValuesRemainUnallocatedInRegionMemoryPlan() {
        GraphValueRef virtualValue = GraphValueRef.node(31);
        GraphValueRef materializedValue = GraphValueRef.node(32);
        Tensor a = Tensor.scalar(1.0f);
        Tensor b = a.relu();

        OptimizedRegion region = new OptimizedRegion(
                "region-virtual",
                partition("region-virtual", PartitionTarget.CPU, List.of(30, 31, 32), List.of(0), List.of(GraphValueRef.node(32)), List.of(GraphValueRef.node(32))),
                PartitionTarget.CPU,
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
                                RegionOptimizationTrace.empty()
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
                                RegionOptimizationTrace.empty()
                        )
                ),
                List.of(
                        new RegionValue(
                                virtualValue,
                                GraphValueRef.node(31),
                                31,
                                1,
                                ValueTransportKind.VIRTUAL,
                                ValueTypeContract.same(DataType.FLOAT32),
                                false
                        ),
                        new RegionValue(
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
                RegionOptimizationTrace.empty()
        );

        List<Tensor> graph = b.topologicalSort();
        MemoryPlan plan = planFor(graph, List.of(region), graph.indexOf(b));

        assertNotNull(plan);
        assertEquals(MaterializationDecision.VIRTUALIZE, plan.materializationPlanOf(virtualValue).decision());
        assertFalse(plan.materializationPlanOf(virtualValue).allocatesStorage());
        assertEquals(RegionMemoryBindingKind.NONE, plan.regionMemoryBindingOf(virtualValue).kind());
        assertEquals(MaterializationDecision.MATERIALIZE, plan.materializationPlanOf(materializedValue).decision());
        assertEquals(materializedValue, plan.graphValueRefOfNodeId(materializedValue.nodeId()));
        assertTrue(plan.regionMemoryBindingOf(materializedValue).hasBindingId());
        assertEquals(plan.regionSlotIdOf(materializedValue), plan.runtimeSlotIdOfNodeId(materializedValue.nodeId()));
        assertEquals(1, plan.regionSlotSize(plan.regionSlotIdOf(materializedValue)));
    }

    private static MemoryPlan planFor(
            List<Tensor> graph,
            List<OptimizedRegion> regions,
            int forwardBoundaryNodeId
    ) {
        return MemoryPlanner.plan(
                new MemoryPlanningInput(
                        CompiledNodeSnapshotter.snapshot(graph, BackendIntentPlan.empty()),
                        regions,
                        Map.of(),
                        ExecutionMode.FORWARD,
                        false,
                        forwardBoundaryNodeId
                ),
                MemoryPlannerPolicy.defaults()
        );
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
