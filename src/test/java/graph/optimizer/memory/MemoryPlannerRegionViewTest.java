package graph.optimizer.memory;

import backend.runtime.ExecutionMode;
import config.optimizer.FuseConfig;
import config.compile.CompileConfig;
import graph.CompiledGraph;
import graph.CompiledNode;
import graph.execution.trace.PartitionDecisionTrace;
import graph.optimizer.partition.Partition;
import graph.optimizer.partition.PartitionBoundaryReason;
import graph.optimizer.partition.PartitionEdge;
import graph.optimizer.partition.PartitionPlannerStrategy;
import graph.optimizer.partition.PartitionTarget;
import graph.optimizer.partition.PartitionValue;
import graph.optimizer.partition.PartitionValueRef;
import graph.optimizer.region.DefaultRegionOptimizer;
import graph.optimizer.region.ExecutionUnit;
import graph.optimizer.region.ExecutionUnitKind;
import graph.optimizer.region.MaterializationDecision;
import graph.optimizer.region.OptimizedRegion;
import graph.optimizer.region.RegionOptimizationTrace;
import graph.optimizer.region.RegionOptimizationContext;
import graph.optimizer.region.RegionValue;
import graph.optimizer.region.RegionValueRef;
import graph.optimizer.region.ValueTransportKind;
import graph.optimizer.region.ValueTypeContract;
import graph.optimizer.memory.MemoryOptimizerRule;
import graph.optimizer.state.OptimizerState;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import java.lang.reflect.Method;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
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
        List<CompiledNode> compiledNodes = CompiledNode.snapshot(graph);
        Partition partition = partition(
                "cpu-cont",
                PartitionTarget.CPU,
                List.of(2, 3),
                List.of(0, 1),
                List.of(PartitionValueRef.ofNode(3)),
                List.of()
        );
        OptimizedRegion region = new DefaultRegionOptimizer().optimize(
                partition,
                new RegionOptimizationContext(compiledNodes, FuseConfig.inferenceDefaults())
        );

        OptimizerState state = OptimizerState.ofGraph(graph, out)
                .withExecutionMetadata(ExecutionMode.FORWARD, false, graph.indexOf(out))
                .withPartitions(List.of(partition))
                .withOptimizedRegions(List.of(region));

        OptimizerState planned = new MemoryOptimizerRule().apply(state);

        assertNotNull(planned.memoryPlan());
        assertEquals(1, planned.memoryPlan().structuralView().optimizedRegionIds().size());
        assertEquals(1, planned.memoryPlan().structuralView().continuationValues().size());
        RegionValueRef outputRef = region.regionValues().stream()
                .filter(value -> value.transportKind() == ValueTransportKind.CONTINUATION)
                .map(RegionValue::ref)
                .findFirst()
                .orElseThrow();
        assertEquals(outputRef, planned.memoryPlan().regionValueRefOf(out));
        assertEquals(MaterializationDecision.CONTINUE, planned.memoryPlan().materializationPlanOf(outputRef).decision());
        assertEquals(RegionMemoryBindingKind.CONTINUATION, planned.memoryPlan().regionMemoryBindingOf(outputRef).kind());
        assertEquals(planned.memoryPlan().regionSlotIdOf(outputRef), planned.memoryPlan().runtimeSlotIdOf(out));
        assertNotNull(planned.memoryPlan().regionSlotIdOf(outputRef));
        assertEquals(4, planned.memoryPlan().regionSlotSize(planned.memoryPlan().regionSlotIdOf(outputRef)));
        assertTrue(planned.memoryPlan().explain().contains("=== Structural Memory View ==="));
        assertTrue(planned.memoryPlan().explain().contains("=== Region Value Plan ==="));
    }

    @Test
    void memoryPlannerCapturesCrossRegionValueFlow() {
        RegionValueRef shared = RegionValueRef.ofNode(11);
        RegionValueRef finalOut = RegionValueRef.ofNode(21);
        Tensor a = Tensor.scalar(1.0f);
        Tensor b = Tensor.scalar(2.0f);
        Tensor out = a.add(b);

        OptimizedRegion producer = new OptimizedRegion(
                "region-a",
                partition("region-a", PartitionTarget.CPU, List.of(10, 11), List.of(0), List.of(PartitionValueRef.ofNode(11)), List.of()),
                PartitionTarget.CPU,
                List.of(new ExecutionUnit(
                        "unit-a",
                        ExecutionUnitKind.SINGLE_OP,
                        PartitionTarget.CPU,
                        List.of(RegionValueRef.ofNode(0)),
                        List.of(shared),
                        List.of(),
                        List.of(shared),
                        List.of(10, 11),
                        2L,
                        List.of(0),
                        RegionOptimizationTrace.empty()
                )),
                List.of(new RegionValue(
                        shared,
                        PartitionValueRef.ofNode(11),
                        out,
                        11,
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
                partition("region-b", PartitionTarget.CPU, List.of(20, 21), List.of(11), List.of(PartitionValueRef.ofNode(21)), List.of(PartitionValueRef.ofNode(21))),
                PartitionTarget.CPU,
                List.of(new ExecutionUnit(
                        "unit-b",
                        ExecutionUnitKind.SINGLE_OP,
                        PartitionTarget.CPU,
                        List.of(shared),
                        List.of(finalOut),
                        List.of(finalOut),
                        List.of(),
                        List.of(20, 21),
                        2L,
                        List.of(11),
                        RegionOptimizationTrace.empty()
                )),
                List.of(new RegionValue(
                        finalOut,
                        PartitionValueRef.ofNode(21),
                        out,
                        21,
                        1,
                        ValueTransportKind.MATERIALIZED,
                        ValueTypeContract.same(DataType.FLOAT32),
                        true
                )),
                List.of(finalOut),
                RegionOptimizationTrace.empty()
        );

        List<Tensor> graph = out.topologicalSort();
        OptimizerState state = OptimizerState.ofGraph(graph, out)
                .withExecutionMetadata(ExecutionMode.FORWARD, false, graph.indexOf(out))
                .withOptimizedRegions(List.of(producer, consumer));

        OptimizerState planned = new MemoryOptimizerRule().apply(state);

        assertNotNull(planned.memoryPlan());
        assertEquals(1, planned.memoryPlan().structuralView().crossRegionDependencyCount());
        var flow = planned.memoryPlan().structuralView().flowOf(shared);
        assertNotNull(flow);
        assertEquals(MaterializationDecision.CONTINUE, flow.decision());
        assertEquals(List.of("region-b"), flow.consumerRegionIds());
        assertEquals(List.of("unit-b"), flow.consumerUnitIds());
        assertEquals(finalOut, planned.memoryPlan().regionValueRefOf(out));
        assertEquals(0, planned.memoryPlan().regionValueLifetimeOf(shared).birthStep());
        assertEquals(graph.size(), planned.memoryPlan().regionValueLifetimeOf(shared).lastUseStep());
        assertTrue(planned.memoryPlan().regionValueLifetimeOf(shared).isCrossRegion());
        assertEquals(RegionMemoryBindingKind.CONTINUATION, planned.memoryPlan().regionMemoryBindingOf(shared).kind());
        assertEquals(planned.memoryPlan().regionSlotIdOf(finalOut), planned.memoryPlan().runtimeSlotIdOf(out));
        assertEquals(1, planned.memoryPlan().regionSlotSize(planned.memoryPlan().regionSlotIdOf(shared)));
        assertEquals(1, planned.memoryPlan().handoffRequirements().size());
    }

    @Test
    void terminalGradientTargetsDoNotReuseRegionBindings() throws Exception {
        Tensor a = new Tensor(new double[]{1, 5, 3}, new int[]{3}, null, "a", DataType.FLOAT64);
        Tensor b = new Tensor(new double[]{2, 4, 3}, new int[]{3}, null, "b", DataType.FLOAT64);
        a.setRequiresGrad(true);
        b.setRequiresGrad(true);

        Tensor min = a.minimum(b);
        CompiledGraph compiled = CompiledGraph.compile(min, CompileConfig.training());

        Method compiledMemoryPlan = CompiledGraph.class.getDeclaredMethod("compiledMemoryPlan");
        compiledMemoryPlan.setAccessible(true);
        MemoryPlan plan = (MemoryPlan) compiledMemoryPlan.invoke(compiled);

        assertNotNull(plan);
        List<RegionValueRef> materializedValues = plan.structuralView().materializedValues();
        assertEquals(2, materializedValues.size());
        assertTrue(materializedValues.stream()
                .allMatch(valueRef -> plan.regionValueLifetimeOf(valueRef).lastUseStep() == compiled.getCompiledGraphAsList().size()));
        long distinctSlotCount = materializedValues.stream()
                .map(plan::regionSlotIdOf)
                .distinct()
                .count();
        assertEquals(materializedValues.size(), distinctSlotCount);
    }

    @Test
    void memoryOptimizerRuleKeepsStructuralMemoryPlanForBfloat16Graphs() {
        Tensor a = new Tensor(new short[]{1, 2, 3, 4}, new int[]{4}, null, "a", DataType.BFLOAT16);
        Tensor b = new Tensor(new short[]{5, 6, 7, 8}, new int[]{4}, null, "b", DataType.BFLOAT16);
        Tensor add = a.add(b);
        Tensor out = add.relu();

        List<Tensor> graph = out.topologicalSort();
        List<CompiledNode> compiledNodes = CompiledNode.snapshot(graph);
        Partition partition = partition(
                "bf16-region",
                PartitionTarget.CPU,
                List.of(2, 3),
                List.of(0, 1),
                List.of(PartitionValueRef.ofNode(3)),
                List.of()
        );
        OptimizedRegion region = new DefaultRegionOptimizer().optimize(
                partition,
                new RegionOptimizationContext(compiledNodes, FuseConfig.inferenceDefaults())
        );

        OptimizerState state = OptimizerState.ofGraph(graph, out)
                .withExecutionMetadata(ExecutionMode.FORWARD, false, graph.indexOf(out))
                .withPartitions(List.of(partition))
                .withOptimizedRegions(List.of(region));

        OptimizerState planned = new MemoryOptimizerRule().apply(state);

        assertNotNull(planned.memoryPlan());
        assertEquals(1, planned.memoryPlan().structuralView().optimizedRegionIds().size());
        assertSame(planned.memoryPlan(), MemoryOptimizerRule.lastPlan());
    }

    @Test
    void virtualValuesRemainUnallocatedInRegionMemoryPlan() {
        RegionValueRef virtualValue = RegionValueRef.ofNode(31);
        RegionValueRef materializedValue = RegionValueRef.ofNode(32);
        Tensor a = Tensor.scalar(1.0f);
        Tensor b = a.relu();

        OptimizedRegion region = new OptimizedRegion(
                "region-virtual",
                partition("region-virtual", PartitionTarget.CPU, List.of(30, 31, 32), List.of(0), List.of(PartitionValueRef.ofNode(32)), List.of(PartitionValueRef.ofNode(32))),
                PartitionTarget.CPU,
                List.of(
                        new ExecutionUnit(
                                "unit-virtual",
                                ExecutionUnitKind.SINGLE_OP,
                                PartitionTarget.CPU,
                                List.of(RegionValueRef.ofNode(0)),
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
                                PartitionValueRef.ofNode(31),
                                a,
                                31,
                                1,
                                ValueTransportKind.VIRTUAL,
                                ValueTypeContract.same(DataType.FLOAT32),
                                false
                        ),
                        new RegionValue(
                                materializedValue,
                                PartitionValueRef.ofNode(32),
                                b,
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
        OptimizerState state = OptimizerState.ofGraph(graph, b)
                .withExecutionMetadata(ExecutionMode.FORWARD, false, graph.indexOf(b))
                .withOptimizedRegions(List.of(region));

        OptimizerState planned = new MemoryOptimizerRule().apply(state);

        assertNotNull(planned.memoryPlan());
        assertEquals(MaterializationDecision.VIRTUALIZE, planned.memoryPlan().materializationPlanOf(virtualValue).decision());
        assertFalse(planned.memoryPlan().materializationPlanOf(virtualValue).allocatesStorage());
        assertEquals(RegionMemoryBindingKind.NONE, planned.memoryPlan().regionMemoryBindingOf(virtualValue).kind());
        assertEquals(MaterializationDecision.MATERIALIZE, planned.memoryPlan().materializationPlanOf(materializedValue).decision());
        assertEquals(materializedValue, planned.memoryPlan().regionValueRefOf(b));
        assertTrue(planned.memoryPlan().regionMemoryBindingOf(materializedValue).hasBindingId());
        assertEquals(planned.memoryPlan().regionSlotIdOf(materializedValue), planned.memoryPlan().runtimeSlotIdOf(b));
        assertEquals(1, planned.memoryPlan().regionSlotSize(planned.memoryPlan().regionSlotIdOf(materializedValue)));
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
