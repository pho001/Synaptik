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
import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
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
    void gpuRegionBuildsFusedElementwiseSubchainWithoutShorteningRegion() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "gpuSubchainA", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 0f, 0f, 1f, 1f, 1f}, new int[]{3, 2}, null, "gpuSubchainB", DataType.FLOAT32);
        Tensor bias = new Tensor(new float[]{0.25f, -0.5f}, new int[]{2}, null, "gpuSubchainBias", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor add = matmul.add(bias);
        Tensor relu = add.relu();
        Tensor out = relu.exp();

        List<CompiledNode> nodes = CompiledNode.snapshot(out.topologicalSort());
        int matmulNodeId = nodeId(nodes, Operation.OpType.MATMUL);
        int addNodeId = nodeId(nodes, Operation.OpType.ADD);
        int reluNodeId = nodeId(nodes, Operation.OpType.RELU);
        int expNodeId = nodeId(nodes, Operation.OpType.EXP);
        List<Integer> selectedNodeIds = List.of(matmulNodeId, addNodeId, reluNodeId, expNodeId);
        Partition partition = partition(
                "gpu-elementwise-subchain",
                PartitionTarget.GPU_METAL,
                selectedNodeIds,
                externalInputNodeIds(nodes, selectedNodeIds),
                List.of(PartitionValueRef.ofNode(expNodeId)),
                List.of(PartitionValueRef.ofNode(expNodeId))
        );

        OptimizedRegion region = new DefaultRegionOptimizer().optimize(
                partition,
                new RegionOptimizationContext(nodes, FuseConfig.inferenceDefaults())
        );

        assertEquals(selectedNodeIds, region.sourcePartition().orderedNodeIds());
        assertTrue(region.executionUnits().stream().anyMatch(unit -> unit.kind() == ExecutionUnitKind.FUSED_ELEMENTWISE));
        ExecutionUnit fused = region.executionUnits().stream()
                .filter(unit -> unit.kind() == ExecutionUnitKind.FUSED_ELEMENTWISE)
                .findFirst()
                .orElseThrow();
        assertEquals(List.of(addNodeId, reluNodeId, expNodeId), fused.orderedNodeIds());
        assertTrue(fused.trace().events().stream().anyMatch(message -> message.contains("fused-subchain:")));
        assertFalse(region.regionValues().stream()
                .filter(value -> value.producerNodeId() == addNodeId || value.producerNodeId() == reluNodeId)
                .anyMatch(value -> value.transportKind() == ValueTransportKind.MATERIALIZED));
    }

    @Test
    void gpuRegionKeepsMixedMatmulAndElementwiseRegionSelected() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "gpuMixedA", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 0f, 0f, 1f, 1f, 1f}, new int[]{3, 2}, null, "gpuMixedB", DataType.FLOAT32);
        Tensor bias = new Tensor(new float[]{0.25f, -0.5f}, new int[]{2}, null, "gpuMixedBias", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor add = matmul.add(bias);
        Tensor relu = add.relu();
        Tensor out = relu.exp();

        List<CompiledNode> nodes = CompiledNode.snapshot(out.topologicalSort());
        int matmulNodeId = nodeId(nodes, Operation.OpType.MATMUL);
        int addNodeId = nodeId(nodes, Operation.OpType.ADD);
        int reluNodeId = nodeId(nodes, Operation.OpType.RELU);
        int expNodeId = nodeId(nodes, Operation.OpType.EXP);
        List<Integer> selectedNodeIds = List.of(matmulNodeId, addNodeId, reluNodeId, expNodeId);
        Partition partition = partition(
                "cuda-mixed-subchain",
                PartitionTarget.GPU_CUDA,
                selectedNodeIds,
                externalInputNodeIds(nodes, selectedNodeIds),
                List.of(PartitionValueRef.ofNode(expNodeId)),
                List.of(PartitionValueRef.ofNode(expNodeId))
        );

        OptimizedRegion region = new DefaultRegionOptimizer().optimize(
                partition,
                new RegionOptimizationContext(nodes, FuseConfig.inferenceDefaults())
        );

        assertEquals(selectedNodeIds, region.sourcePartition().orderedNodeIds());
        assertEquals(selectedNodeIds, region.executionUnits().stream()
                .flatMap(unit -> unit.orderedNodeIds().stream())
                .toList());
        assertEquals(ExecutionUnitKind.UNIT_KERNEL, region.executionUnits().getFirst().kind());
        assertEquals(ExecutionUnitKind.FUSED_ELEMENTWISE, region.executionUnits().get(1).kind());
        assertEquals(List.of(addNodeId, reluNodeId, expNodeId), region.executionUnits().get(1).orderedNodeIds());
    }

    @Test
    void gpuRegionBuildsMatmulBiasActivationEpilogueUnit() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "gpuEpilogueA", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 0f, 0f, 1f, 1f, 1f}, new int[]{3, 2}, null, "gpuEpilogueB", DataType.FLOAT32);
        Tensor bias = new Tensor(new float[]{0.25f, -0.5f}, new int[]{2}, null, "gpuEpilogueBias", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor add = matmul.add(bias);
        Tensor out = add.relu();

        List<CompiledNode> nodes = CompiledNode.snapshot(out.topologicalSort());
        int matmulNodeId = nodeId(nodes, Operation.OpType.MATMUL);
        int addNodeId = nodeId(nodes, Operation.OpType.ADD);
        int reluNodeId = nodeId(nodes, Operation.OpType.RELU);
        List<Integer> selectedNodeIds = List.of(matmulNodeId, addNodeId, reluNodeId);
        Partition partition = partition(
                "gpu-epilogue-subregion",
                PartitionTarget.GPU_METAL,
                selectedNodeIds,
                externalInputNodeIds(nodes, selectedNodeIds),
                List.of(PartitionValueRef.ofNode(reluNodeId)),
                List.of(PartitionValueRef.ofNode(reluNodeId))
        );

        OptimizedRegion region = new DefaultRegionOptimizer().optimize(
                partition,
                new RegionOptimizationContext(nodes, FuseConfig.inferenceDefaults())
        );

        assertEquals(selectedNodeIds, region.sourcePartition().orderedNodeIds());
        assertEquals(1, region.executionUnits().size());
        ExecutionUnit epilogue = region.executionUnits().getFirst();
        assertEquals(ExecutionUnitKind.SPECIALIZED_PRIMITIVE, epilogue.kind());
        assertEquals(selectedNodeIds, epilogue.orderedNodeIds());
        assertTrue(epilogue.trace().events().stream().anyMatch(message -> message.contains("gpu-epilogue-subregion:")));
        assertFalse(epilogue.requiredPreparedInputNodeIds().contains(addNodeId));
        assertFalse(region.regionValues().stream()
                .filter(value -> value.producerNodeId() == addNodeId)
                .anyMatch(value -> value.transportKind() == ValueTransportKind.MATERIALIZED));
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

    private static int nodeId(List<CompiledNode> nodes, Operation.OpType opType) {
        return nodes.stream()
                .filter(node -> node.operation() != null && node.operation().opType() == opType)
                .map(CompiledNode::id)
                .findFirst()
                .orElseThrow();
    }

    private static List<Integer> externalInputNodeIds(List<CompiledNode> nodes, List<Integer> selectedNodeIds) {
        java.util.Set<Integer> selected = java.util.Set.copyOf(selectedNodeIds);
        java.util.LinkedHashSet<Integer> out = new java.util.LinkedHashSet<>();
        for (int nodeId : selectedNodeIds) {
            CompiledNode node = nodes.stream().filter(candidate -> candidate.id() == nodeId).findFirst().orElseThrow();
            node.inputIds().stream().filter(inputId -> !selected.contains(inputId)).forEach(out::add);
        }
        return List.copyOf(out);
    }
}
