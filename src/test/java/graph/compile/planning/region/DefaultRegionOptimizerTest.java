package graph.compile.planning.region;

import config.optimizer.FuseConfig;
import graph.CompiledNode;
import graph.execution.trace.PartitionDecisionTrace;
import graph.compile.planning.partition.Partition;
import graph.compile.planning.partition.PartitionBoundaryReason;
import graph.compile.planning.partition.PartitionEdge;
import graph.compile.planning.partition.PartitionPlannerStrategy;
import graph.compile.planning.partition.PartitionTarget;
import graph.compile.planning.partition.PartitionValue;
import graph.compile.planning.value.GraphValueRef;
import graph.compile.planning.partition.cost.AcceleratorPartitionScoreModel;
import graph.compile.planning.region.lowering.OperationSemanticClassifier;
import graph.compile.planning.region.lowering.OperationSemanticLevel;
import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import graph.compile.intent.BackendIntentPlan;

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

        List<CompiledNode> nodes = CompiledNode.snapshot(out.topologicalSort(), BackendIntentPlan.empty());
        Partition partition = partition(
                "cpu-eltwise",
                PartitionTarget.CPU,
                List.of(2, 3, 4),
                List.of(0, 1),
                List.of(GraphValueRef.node(4)),
                List.of(GraphValueRef.node(4))
        );

        DefaultRegionOptimizer optimizer = new DefaultRegionOptimizer();
        OptimizedRegion region = optimizer.optimize(partition, new RegionOptimizationContext(nodes, FuseConfig.inferenceDefaults()));

        assertEquals(1, region.executionUnits().size());
        assertEquals(ExecutionUnitKind.FUSED_ELEMENTWISE, region.executionUnits().getFirst().kind());
        assertEquals(ValueTransportKind.MATERIALIZED,
                region.regionValues().stream()
                        .filter(v -> v.ref().equals(GraphValueRef.node(4)))
                        .findFirst().orElseThrow().transportKind());
    }

    @Test
    void mixedCpuPartitionFallsBackToSingleOpUnits() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{7f, 8f, 9f, 10f, 11f, 12f}, new int[]{3, 2}, null, "b", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor out = matmul.relu();

        List<CompiledNode> nodes = CompiledNode.snapshot(out.topologicalSort(), BackendIntentPlan.empty());
        Partition partition = partition(
                "cpu-mixed",
                PartitionTarget.CPU,
                List.of(2, 3),
                List.of(0, 1),
                List.of(GraphValueRef.node(3)),
                List.of(GraphValueRef.node(3))
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

        List<CompiledNode> nodes = CompiledNode.snapshot(out.topologicalSort(), BackendIntentPlan.empty());
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
                List.of(GraphValueRef.node(expNodeId)),
                List.of(GraphValueRef.node(expNodeId))
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

        List<CompiledNode> nodes = CompiledNode.snapshot(out.topologicalSort(), BackendIntentPlan.empty());
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
                List.of(GraphValueRef.node(expNodeId)),
                List.of(GraphValueRef.node(expNodeId))
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

        List<CompiledNode> nodes = CompiledNode.snapshot(out.topologicalSort(), BackendIntentPlan.empty());
        int matmulNodeId = nodeId(nodes, Operation.OpType.MATMUL);
        int addNodeId = nodeId(nodes, Operation.OpType.ADD);
        int reluNodeId = nodeId(nodes, Operation.OpType.RELU);
        List<Integer> selectedNodeIds = List.of(matmulNodeId, addNodeId, reluNodeId);
        Partition partition = partition(
                "gpu-epilogue-subregion",
                PartitionTarget.GPU_METAL,
                selectedNodeIds,
                externalInputNodeIds(nodes, selectedNodeIds),
                List.of(GraphValueRef.node(reluNodeId)),
                List.of(GraphValueRef.node(reluNodeId))
        );

        OptimizedRegion region = new DefaultRegionOptimizer().optimize(
                partition,
                new RegionOptimizationContext(nodes, FuseConfig.inferenceDefaults())
        );

        assertEquals(selectedNodeIds, region.sourcePartition().orderedNodeIds());
        assertEquals(2, region.executionUnits().size());
        ExecutionUnit matmulUnit = region.executionUnits().getFirst();
        ExecutionUnit fusedEpilogueTail = region.executionUnits().get(1);
        assertEquals(ExecutionUnitKind.UNIT_KERNEL, matmulUnit.kind());
        assertEquals(List.of(matmulNodeId), matmulUnit.orderedNodeIds());
        assertEquals(ExecutionUnitKind.FUSED_ELEMENTWISE, fusedEpilogueTail.kind());
        assertEquals(List.of(addNodeId, reluNodeId), fusedEpilogueTail.orderedNodeIds());
        assertTrue(fusedEpilogueTail.trace().events().stream().anyMatch(message -> message.contains("fused-subchain:")));
        assertTrue(fusedEpilogueTail.trace().events().stream().anyMatch(message -> message.contains("region-unit-node:")));
        assertFalse(fusedEpilogueTail.requiredPreparedInputNodeIds().contains(addNodeId));
        assertFalse(region.regionValues().stream()
                .filter(value -> value.producerNodeId() == addNodeId)
                .anyMatch(value -> value.transportKind() == ValueTransportKind.MATERIALIZED));
    }

    @Test
    void gpuRegionDoesNotUseCpuFusedNodeForRegionInternalFusion() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{4}, null, "gpuNoCpuFusedA", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{5f, 6f, 7f, 8f}, new int[]{4}, null, "gpuNoCpuFusedB", DataType.FLOAT32);
        Tensor mul = a.mul(b);
        Tensor tanh = mul.tanh();
        Tensor out = tanh.sigmoid();

        List<CompiledNode> nodes = CompiledNode.snapshot(out.topologicalSort(), BackendIntentPlan.empty());
        int mulNodeId = nodeId(nodes, Operation.OpType.MUL);
        int tanhNodeId = nodeId(nodes, Operation.OpType.TANH);
        int sigmoidNodeId = nodeId(nodes, Operation.OpType.SIGMOID);
        List<Integer> selectedNodeIds = List.of(mulNodeId, tanhNodeId, sigmoidNodeId);
        Partition partition = partition(
                "gpu-no-cpu-fused",
                PartitionTarget.GPU_METAL,
                selectedNodeIds,
                externalInputNodeIds(nodes, selectedNodeIds),
                List.of(GraphValueRef.node(sigmoidNodeId)),
                List.of(GraphValueRef.node(sigmoidNodeId))
        );

        OptimizedRegion region = new DefaultRegionOptimizer().optimize(
                partition,
                new RegionOptimizationContext(nodes, FuseConfig.inferenceDefaults())
        );

        assertEquals(1, region.executionUnits().size());
        ExecutionUnit unit = region.executionUnits().getFirst();
        assertEquals(ExecutionUnitKind.FUSED_ELEMENTWISE, unit.kind());
        assertTrue(unit.trace().events().stream().anyMatch(message -> message.contains("region-unit-node:")));
        assertFalse(unit.orderedNodeIds().stream()
                .map(nodes::get)
                .map(CompiledNode::operation)
                .anyMatch(operation -> operation != null && operation.opType() == Operation.OpType.FUSED));
    }

    @Test
    void operationSemanticClassifierKeepsHighLevelOpsVisibleForRegionPolicy() {
        assertEquals(OperationSemanticLevel.CANONICAL_HIGH_LEVEL,
                OperationSemanticClassifier.classify(Operation.OpType.LINEAR));
        assertEquals(OperationSemanticLevel.BACKEND_FRIENDLY_HIGH_LEVEL,
                OperationSemanticClassifier.classify(Operation.OpType.CONV2D));
        assertEquals(OperationSemanticLevel.BACKEND_FRIENDLY_HIGH_LEVEL,
                OperationSemanticClassifier.classify(Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION));
        assertEquals(OperationSemanticLevel.TRAINING_BACKWARD,
                OperationSemanticClassifier.classify(Operation.OpType.LOG_SOFTMAX_GRAD));
        assertEquals(OperationSemanticLevel.FUSED,
                OperationSemanticClassifier.classify(Operation.OpType.FUSED));
    }

    @Test
    void partitionOutputThatIsNotRequiredMaterializedBecomesContinuationValue() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{4}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{5f, 6f, 7f, 8f}, new int[]{4}, null, "b", DataType.FLOAT32);
        Tensor add = a.add(b);
        Tensor out = add.relu();

        List<CompiledNode> nodes = CompiledNode.snapshot(out.topologicalSort(), BackendIntentPlan.empty());
        Partition partition = partition(
                "cpu-cont",
                PartitionTarget.CPU,
                List.of(2, 3),
                List.of(0, 1),
                List.of(GraphValueRef.node(3)),
                List.of()
        );

        DefaultRegionOptimizer optimizer = new DefaultRegionOptimizer();
        OptimizedRegion region = optimizer.optimize(partition, new RegionOptimizationContext(nodes, FuseConfig.inferenceDefaults()));

        RegionValue output = region.regionValues().stream()
                .filter(v -> v.ref().equals(GraphValueRef.node(3)))
                .findFirst()
                .orElseThrow();

        assertEquals(ValueTransportKind.CONTINUATION, output.transportKind());
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
