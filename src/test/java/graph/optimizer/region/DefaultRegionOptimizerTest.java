package graph.optimizer.region;

import config.optimizer.FuseConfig;
import graph.CompiledNode;
import graph.execution.trace.PartitionDecisionTrace;
import graph.optimizer.partition.Partition;
import graph.optimizer.partition.PartitionBoundaryReason;
import graph.optimizer.partition.PartitionEdge;
import graph.optimizer.partition.PartitionPlannerStrategy;
import graph.optimizer.partition.PartitionTarget;
import graph.optimizer.partition.PartitionValue;
import graph.optimizer.partition.PartitionValueRef;
import graph.optimizer.partition.cost.AcceleratorPartitionScoreModel;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class DefaultRegionOptimizerTest {
    @Test
    void fusesSimpleCpuElementwisePartitionIntoSingleExecutionUnit() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{4}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{5f, 6f, 7f, 8f}, new int[]{4}, null, "b", DataType.FLOAT32);
        Tensor add = a.add(b);
        Tensor relu = add.relu();
        Tensor out = relu.tanh();

        List<CompiledNode> nodes = CompiledNode.snapshot(out.topologicalSort());
        Partition partition = partition(
                "cpu-eltwise",
                PartitionTarget.CPU,
                List.of(2, 3, 4),
                List.of(0, 1),
                List.of(PartitionValueRef.ofNode(4)),
                List.of(PartitionValueRef.ofNode(4))
        );

        DefaultRegionOptimizer optimizer = new DefaultRegionOptimizer();
        OptimizedRegion region = optimizer.optimize(partition, new RegionOptimizationContext(nodes, FuseConfig.inferenceDefaults()));

        assertEquals(1, region.executionUnits().size());
        assertEquals(ExecutionUnitKind.FUSED_ELEMENTWISE, region.executionUnits().getFirst().kind());
        assertEquals(ValueTransportKind.MATERIALIZED,
                region.regionValues().stream()
                        .filter(v -> v.ref().equals(RegionValueRef.ofNode(4)))
                        .findFirst().orElseThrow().transportKind());
    }

    @Test
    void mixedCpuPartitionFallsBackToSingleOpUnits() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{7f, 8f, 9f, 10f, 11f, 12f}, new int[]{3, 2}, null, "b", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor out = matmul.relu();

        List<CompiledNode> nodes = CompiledNode.snapshot(out.topologicalSort());
        Partition partition = partition(
                "cpu-mixed",
                PartitionTarget.CPU,
                List.of(2, 3),
                List.of(0, 1),
                List.of(PartitionValueRef.ofNode(3)),
                List.of(PartitionValueRef.ofNode(3))
        );

        DefaultRegionOptimizer optimizer = new DefaultRegionOptimizer();
        OptimizedRegion region = optimizer.optimize(partition, new RegionOptimizationContext(nodes, FuseConfig.inferenceDefaults()));

        assertEquals(2, region.executionUnits().size());
        assertTrue(region.executionUnits().stream().allMatch(unit -> unit.kind() == ExecutionUnitKind.UNIT_KERNEL));
    }

    @Test
    void partitionOutputThatIsNotRequiredMaterializedBecomesContinuationValue() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{4}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{5f, 6f, 7f, 8f}, new int[]{4}, null, "b", DataType.FLOAT32);
        Tensor add = a.add(b);
        Tensor out = add.relu();

        List<CompiledNode> nodes = CompiledNode.snapshot(out.topologicalSort());
        Partition partition = partition(
                "cpu-cont",
                PartitionTarget.CPU,
                List.of(2, 3),
                List.of(0, 1),
                List.of(PartitionValueRef.ofNode(3)),
                List.of()
        );

        DefaultRegionOptimizer optimizer = new DefaultRegionOptimizer();
        OptimizedRegion region = optimizer.optimize(partition, new RegionOptimizationContext(nodes, FuseConfig.inferenceDefaults()));

        RegionValue output = region.regionValues().stream()
                .filter(v -> v.ref().equals(RegionValueRef.ofNode(3)))
                .findFirst()
                .orElseThrow();

        assertEquals(ValueTransportKind.CONTINUATION, output.transportKind());
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
                new AcceleratorPartitionScoreModel.CandidateMetrics(orderedNodeIds.size(), internalEdges.size(), externalInputNodeIds.size(), 0, Math.max(0, orderedNodeIds.size() - 1)),
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
