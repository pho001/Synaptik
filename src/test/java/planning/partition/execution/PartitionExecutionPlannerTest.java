package planning.partition.execution;

import config.optimizer.FuseConfig;
import config.compile.CompileConfig;
import graph.compile.CompiledNodeSnapshotter;
import graph.model.CompiledNode;
import graph.CompiledGraph;
import trace.compile.PartitionDecisionTrace;
import planning.partition.Partition;
import planning.partition.PartitionBoundaryReason;
import planning.partition.PartitionEdge;
import planning.partition.PartitionPlannerStrategy;
import planning.partition.PartitionTarget;
import planning.partition.PartitionValue;
import planning.value.GraphValueRef;
import planning.partition.cost.AcceleratorPartitionScoreModel;
import planning.partition.execution.lowering.OperationSemanticClassifier;
import planning.partition.execution.lowering.OperationSemanticLevel;
import planning.partition.specialization.PartitionSpecializationDecision;
import planning.partition.specialization.PartitionSpecializationKind;
import planning.partition.specialization.SdpaBackwardOutputKind;
import planning.partition.specialization.SdpaBackwardSpecializationPayload;
import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import planning.intent.BackendIntentPlan;
import tensor.options.AttentionOptions;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PartitionExecutionPlannerTest {
    @Test
    void fusesSimpleCpuElementwisePartitionIntoSingleExecutionUnit() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{4}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{5f, 6f, 7f, 8f}, new int[]{4}, null, "b", DataType.FLOAT32);
        Tensor add = a.add(b);
        Tensor relu = add.relu();
        Tensor out = relu.tanh();

        List<CompiledNode> nodes = CompiledNodeSnapshotter.snapshot(out.topologicalSort(), BackendIntentPlan.empty());
        Partition partition = partition(
                "cpu-eltwise",
                PartitionTarget.CPU,
                List.of(2, 3, 4),
                List.of(0, 1),
                List.of(GraphValueRef.node(4)),
                List.of(GraphValueRef.node(4))
        );

        PartitionExecutionPlanner planner = new PartitionExecutionPlanner();
        PartitionExecutionPlan executionPlan = planner.planPartition(partition, new PartitionExecutionPlanningContext(nodes, FuseConfig.inferenceDefaults()));

        assertEquals(1, executionPlan.executionUnits().size());
        assertEquals(ExecutionUnitKind.FUSED_ELEMENTWISE, executionPlan.executionUnits().getFirst().kind());
        assertEquals(ValueTransportKind.MATERIALIZED,
                executionPlan.executionValues().stream()
                        .filter(v -> v.ref().equals(GraphValueRef.node(4)))
                        .findFirst().orElseThrow().transportKind());
    }

    @Test
    void cpuMatmulReluPartitionBuildsSpecializedPrimitiveByDefault() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{7f, 8f, 9f, 10f, 11f, 12f}, new int[]{3, 2}, null, "b", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor out = matmul.relu();

        List<CompiledNode> nodes = CompiledNodeSnapshotter.snapshot(out.topologicalSort(), BackendIntentPlan.empty());
        Partition partition = partition(
                "cpu-mixed",
                PartitionTarget.CPU,
                List.of(2, 3),
                List.of(0, 1),
                List.of(GraphValueRef.node(3)),
                List.of(GraphValueRef.node(3))
        );

        PartitionExecutionPlanner planner = new PartitionExecutionPlanner();
        PartitionExecutionPlan executionPlan = planner.planPartition(partition, new PartitionExecutionPlanningContext(nodes, FuseConfig.inferenceDefaults()));

        assertEquals(1, executionPlan.executionUnits().size());
        ExecutionUnit unit = executionPlan.executionUnits().getFirst();
        assertEquals(ExecutionUnitKind.SPECIALIZED_PRIMITIVE, unit.kind());
        assertEquals(PartitionSpecializationKind.MATMUL_RELU, unit.specialization().kind());
        assertEquals(List.of(2, 3), unit.orderedNodeIds());
        assertEquals(List.of(GraphValueRef.node(0), GraphValueRef.node(1)), unit.inputValueRefs());
        assertEquals(List.of(GraphValueRef.node(3)), unit.outputValueRefs());
    }

    @Test
    void cpuMatmulBiasReluPartitionBuildsSpecializedPrimitiveByDefault() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "cpuBiasA", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{7f, 8f, 9f, 10f, 11f, 12f}, new int[]{3, 2}, null, "cpuBiasB", DataType.FLOAT32);
        Tensor bias = new Tensor(new float[]{0.25f, -0.5f}, new int[]{2}, null, "cpuBias", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor add = matmul.add(bias);
        Tensor out = add.relu();

        List<CompiledNode> nodes = CompiledNodeSnapshotter.snapshot(out.topologicalSort(), BackendIntentPlan.empty());
        int matmulNodeId = nodeId(nodes, Operation.OpType.MATMUL);
        int addNodeId = nodeId(nodes, Operation.OpType.ADD);
        int reluNodeId = nodeId(nodes, Operation.OpType.RELU);
        List<Integer> selectedNodeIds = List.of(matmulNodeId, addNodeId, reluNodeId);
        List<Integer> externalInputNodeIds = externalInputNodeIds(nodes, selectedNodeIds);
        Partition partition = partition(
                "cpu-matmul-bias-relu",
                PartitionTarget.CPU,
                selectedNodeIds,
                externalInputNodeIds,
                List.of(GraphValueRef.node(reluNodeId)),
                List.of(GraphValueRef.node(reluNodeId))
        );

        PartitionExecutionPlan executionPlan = new PartitionExecutionPlanner().planPartition(
                partition,
                new PartitionExecutionPlanningContext(nodes, FuseConfig.inferenceDefaults())
        );

        assertEquals(1, executionPlan.executionUnits().size());
        ExecutionUnit unit = executionPlan.executionUnits().getFirst();
        assertEquals(ExecutionUnitKind.SPECIALIZED_PRIMITIVE, unit.kind());
        assertEquals(PartitionSpecializationKind.MATMUL_ADD_BIAS_RELU, unit.specialization().kind());
        assertEquals(selectedNodeIds, unit.orderedNodeIds());
        assertEquals(externalInputNodeIds.stream().map(GraphValueRef::node).toList(), unit.inputValueRefs());
        assertEquals(List.of(GraphValueRef.node(reluNodeId)), unit.outputValueRefs());
        assertTrue(executionPlan.trace().events().stream()
                .anyMatch(event -> event.contains("specialization-candidate-accepted:kind=MATMUL_ADD_BIAS_RELU")));
    }

    @Test
    void cpuLinearBiasPartitionBuildsMatmulAddBiasSpecializedPrimitiveByDefault() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "cpuLinearA", DataType.FLOAT32);
        Tensor weight = new Tensor(new float[]{7f, 8f, 9f, 10f, 11f, 12f}, new int[]{3, 2}, null, "cpuLinearWeight", DataType.FLOAT32);
        Tensor bias = new Tensor(new float[]{0.25f, -0.5f}, new int[]{2}, null, "cpuLinearBias", DataType.FLOAT32);
        Tensor out = a.linear(weight, bias);

        List<CompiledNode> nodes = CompiledNodeSnapshotter.snapshot(out.topologicalSort(), BackendIntentPlan.empty());
        int linearNodeId = nodeId(nodes, Operation.OpType.LINEAR);
        List<Integer> selectedNodeIds = List.of(linearNodeId);
        List<Integer> externalInputNodeIds = externalInputNodeIds(nodes, selectedNodeIds);
        Partition partition = partition(
                "cpu-linear-bias",
                PartitionTarget.CPU,
                selectedNodeIds,
                externalInputNodeIds,
                List.of(GraphValueRef.node(linearNodeId)),
                List.of(GraphValueRef.node(linearNodeId))
        );

        PartitionExecutionPlan executionPlan = new PartitionExecutionPlanner().planPartition(
                partition,
                new PartitionExecutionPlanningContext(nodes, FuseConfig.inferenceDefaults())
        );

        assertEquals(1, executionPlan.executionUnits().size());
        ExecutionUnit unit = executionPlan.executionUnits().getFirst();
        assertEquals(ExecutionUnitKind.SPECIALIZED_PRIMITIVE, unit.kind());
        assertEquals(PartitionSpecializationKind.MATMUL_ADD_BIAS, unit.specialization().kind());
        assertEquals(selectedNodeIds, unit.orderedNodeIds());
        assertEquals(externalInputNodeIds.stream().map(GraphValueRef::node).toList(), unit.inputValueRefs());
        assertEquals(List.of(GraphValueRef.node(linearNodeId)), unit.outputValueRefs());
        assertTrue(executionPlan.trace().events().stream()
                .anyMatch(event -> event.contains("specialization-candidate-accepted:kind=MATMUL_ADD_BIAS")));
    }

    @Test
    void cpuMatmulAddBiasPartitionBuildsMatmulAddBiasSpecializedPrimitiveByDefault() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "cpuMatmulBiasA", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{7f, 8f, 9f, 10f, 11f, 12f}, new int[]{3, 2}, null, "cpuMatmulBiasB", DataType.FLOAT32);
        Tensor bias = new Tensor(new float[]{0.25f, -0.5f}, new int[]{2}, null, "cpuMatmulBias", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor out = matmul.add(bias);

        List<CompiledNode> nodes = CompiledNodeSnapshotter.snapshot(out.topologicalSort(), BackendIntentPlan.empty());
        int matmulNodeId = nodeId(nodes, Operation.OpType.MATMUL);
        int addNodeId = nodeId(nodes, Operation.OpType.ADD);
        List<Integer> selectedNodeIds = List.of(matmulNodeId, addNodeId);
        List<Integer> externalInputNodeIds = externalInputNodeIds(nodes, selectedNodeIds);
        Partition partition = partition(
                "cpu-matmul-add-bias",
                PartitionTarget.CPU,
                selectedNodeIds,
                externalInputNodeIds,
                List.of(GraphValueRef.node(addNodeId)),
                List.of(GraphValueRef.node(addNodeId))
        );

        PartitionExecutionPlan executionPlan = new PartitionExecutionPlanner().planPartition(
                partition,
                new PartitionExecutionPlanningContext(nodes, FuseConfig.inferenceDefaults())
        );

        assertEquals(1, executionPlan.executionUnits().size());
        ExecutionUnit unit = executionPlan.executionUnits().getFirst();
        assertEquals(ExecutionUnitKind.SPECIALIZED_PRIMITIVE, unit.kind());
        assertEquals(PartitionSpecializationKind.MATMUL_ADD_BIAS, unit.specialization().kind());
        assertEquals(selectedNodeIds, unit.orderedNodeIds());
        assertEquals(externalInputNodeIds.stream().map(GraphValueRef::node).toList(), unit.inputValueRefs());
        assertEquals(List.of(GraphValueRef.node(addNodeId)), unit.outputValueRefs());
        assertTrue(executionPlan.trace().events().stream()
                .anyMatch(event -> event.contains("specialization-candidate-accepted:kind=MATMUL_ADD_BIAS")));
    }

    @Test
    void cpuMsePartitionBuildsSpecializedPrimitiveByDefault() {
        Tensor pred = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{4}, null, "defaultMsePred", DataType.FLOAT32);
        Tensor target = new Tensor(new float[]{1.5f, 1f, 2.5f, 3f}, new int[]{4}, null, "defaultMseTarget", DataType.FLOAT32);
        Tensor diff = pred.sub(target);
        Tensor square = diff.mul(diff);
        Tensor loss = square.mean();

        List<CompiledNode> nodes = CompiledNodeSnapshotter.snapshot(loss.topologicalSort(), BackendIntentPlan.empty());
        int diffNodeId = nodeId(nodes, Operation.OpType.SUB);
        int squareNodeId = nodeId(nodes, Operation.OpType.MUL);
        int lossNodeId = nodeId(nodes, Operation.OpType.MEAN);
        List<Integer> selectedNodeIds = List.of(diffNodeId, squareNodeId, lossNodeId);
        Partition partition = partition(
                "cpu-mse-default",
                PartitionTarget.CPU,
                selectedNodeIds,
                externalInputNodeIds(nodes, selectedNodeIds),
                List.of(GraphValueRef.node(lossNodeId)),
                List.of(GraphValueRef.node(lossNodeId))
        );

        PartitionExecutionPlan executionPlan = new PartitionExecutionPlanner().planPartition(
                partition,
                new PartitionExecutionPlanningContext(nodes, FuseConfig.inferenceDefaults())
        );

        assertEquals(1, executionPlan.executionUnits().size());
        ExecutionUnit unit = executionPlan.executionUnits().getFirst();
        assertEquals(ExecutionUnitKind.SPECIALIZED_PRIMITIVE, unit.kind());
        assertEquals(PartitionSpecializationKind.MSE_LOSS, unit.specialization().kind());
        assertTrue(executionPlan.trace().events().stream()
                .anyMatch(event -> event.contains("specialization-candidate-found:kind=MSE_LOSS")));
        assertTrue(executionPlan.trace().events().stream()
                .anyMatch(event -> event.contains("specialization-candidate-accepted:kind=MSE_LOSS")
                        && event.contains("cpu1-mse-loss-executable")));
    }

    @Test
    void cpuSdpaBackwardIsolatedPartitionsBuildSpecializedPrimitiveByDefault() {
        assertSdpaBackwardSpecializationForSingleGradient(SdpaBackwardOutputKind.QUERY, new int[]{1, 2, 2});
        assertSdpaBackwardSpecializationForSingleGradient(SdpaBackwardOutputKind.KEY, new int[]{1, 2, 2});
        assertSdpaBackwardSpecializationForSingleGradient(SdpaBackwardOutputKind.VALUE, new int[]{1, 2, 2});
    }

    @Test
    void acceptedMsePartitionBuildsSpecializedPrimitiveUnit() {
        Tensor pred = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{4}, null, "msePred", DataType.FLOAT32);
        Tensor target = new Tensor(new float[]{1.5f, 1f, 2.5f, 3f}, new int[]{4}, null, "mseTarget", DataType.FLOAT32);
        Tensor diff = pred.sub(target);
        Tensor square = diff.mul(diff);
        Tensor loss = square.mean();

        List<CompiledNode> nodes = CompiledNodeSnapshotter.snapshot(loss.topologicalSort(), BackendIntentPlan.empty());
        int diffNodeId = nodeId(nodes, Operation.OpType.SUB);
        int squareNodeId = nodeId(nodes, Operation.OpType.MUL);
        int lossNodeId = nodeId(nodes, Operation.OpType.MEAN);
        Partition partition = partition(
                "cpu-mse",
                PartitionTarget.CPU,
                List.of(diffNodeId, squareNodeId, lossNodeId),
                externalInputNodeIds(nodes, List.of(diffNodeId, squareNodeId, lossNodeId)),
                List.of(GraphValueRef.node(lossNodeId)),
                List.of(GraphValueRef.node(lossNodeId))
        );

        PartitionExecutionPlan executionPlan = new PartitionExecutionPlanner((partitionTarget, candidate) ->
                partitionTarget == PartitionTarget.CPU && candidate.kind() == PartitionSpecializationKind.MSE_LOSS
                        ? PartitionSpecializationDecision.accept("test-cpu-mse-specialization")
                        : PartitionSpecializationDecision.reject("test-reject")
        ).planPartition(
                partition,
                new PartitionExecutionPlanningContext(nodes, FuseConfig.inferenceDefaults())
        );

        assertEquals(1, executionPlan.executionUnits().size());
        ExecutionUnit unit = executionPlan.executionUnits().getFirst();
        assertEquals(ExecutionUnitKind.SPECIALIZED_PRIMITIVE, unit.kind());
        assertEquals(PartitionSpecializationKind.MSE_LOSS, unit.specialization().kind());
        assertEquals(List.of(diffNodeId, squareNodeId, lossNodeId), unit.orderedNodeIds());
        assertEquals(List.of(GraphValueRef.node(lossNodeId)), unit.outputValueRefs());
        assertEquals(List.of(GraphValueRef.node(lossNodeId)), unit.materializedOutputs());
        assertTrue(unit.virtualOutputs().contains(GraphValueRef.node(diffNodeId)));
        assertTrue(unit.virtualOutputs().contains(GraphValueRef.node(squareNodeId)));
        assertTrue(executionPlan.trace().events().stream()
                .anyMatch(event -> event.contains("specialization-candidate-found:kind=MSE_LOSS")));
        assertTrue(executionPlan.trace().events().stream()
                .anyMatch(event -> event.contains("specialization-candidate-accepted:kind=MSE_LOSS")));
        assertEquals(ValueTransportKind.VIRTUAL, executionPlan.executionValues().stream()
                .filter(value -> value.ref().equals(GraphValueRef.node(diffNodeId)))
                .findFirst().orElseThrow().transportKind());
        assertEquals(ValueTransportKind.VIRTUAL, executionPlan.executionValues().stream()
                .filter(value -> value.ref().equals(GraphValueRef.node(squareNodeId)))
                .findFirst().orElseThrow().transportKind());
    }

    @Test
    void acceptedNestedMeanMsePartitionBuildsSingleSpecializedPrimitiveUnit() {
        Tensor pred = new Tensor(
                new float[]{1f, 2f, 3f, 4f, 5f, 6f},
                new int[]{2, 3},
                null,
                "nestedMsePred",
                DataType.FLOAT32
        );
        Tensor target = new Tensor(
                new float[]{0f, 1f, 2f, 3f, 4f, 5f},
                new int[]{2, 3},
                null,
                "nestedMseTarget",
                DataType.FLOAT32
        );
        Tensor diff = pred.sub(target);
        Tensor square = diff.mul(diff);
        Tensor rowMean = square.mean(1);
        Tensor loss = rowMean.mean(0, true);

        List<Tensor> graph = loss.topologicalSort();
        int diffNodeId = graph.indexOf(diff);
        int squareNodeId = graph.indexOf(square);
        int rowMeanNodeId = graph.indexOf(rowMean);
        int lossNodeId = graph.indexOf(loss);
        List<CompiledNode> nodes = CompiledNodeSnapshotter.snapshot(graph, BackendIntentPlan.empty());
        List<Integer> selectedNodeIds = List.of(diffNodeId, squareNodeId, rowMeanNodeId, lossNodeId);
        Partition partition = partition(
                "cpu-nested-mean-mse",
                PartitionTarget.CPU,
                selectedNodeIds,
                externalInputNodeIds(nodes, selectedNodeIds),
                List.of(GraphValueRef.node(lossNodeId)),
                List.of(GraphValueRef.node(lossNodeId))
        );

        PartitionExecutionPlan executionPlan = new PartitionExecutionPlanner().planPartition(
                partition,
                new PartitionExecutionPlanningContext(nodes, FuseConfig.inferenceDefaults())
        );

        assertEquals(1, executionPlan.executionUnits().size());
        ExecutionUnit unit = executionPlan.executionUnits().getFirst();
        assertEquals(ExecutionUnitKind.SPECIALIZED_PRIMITIVE, unit.kind());
        assertEquals(PartitionSpecializationKind.MSE_LOSS, unit.specialization().kind());
        assertEquals(selectedNodeIds, unit.orderedNodeIds());
        assertTrue(unit.virtualOutputs().contains(GraphValueRef.node(diffNodeId)));
        assertTrue(unit.virtualOutputs().contains(GraphValueRef.node(squareNodeId)));
        assertTrue(unit.virtualOutputs().contains(GraphValueRef.node(rowMeanNodeId)));
        assertTrue(executionPlan.trace().events().stream()
                .anyMatch(event -> event.contains("specialization-candidate-accepted:kind=MSE_LOSS")));
    }

    @Test
    void mixedReductionMsePartitionFallsBackToStructuralUnits() {
        Tensor pred = new Tensor(
                new float[]{1f, 2f, 3f, 4f, 5f, 6f},
                new int[]{2, 3},
                null,
                "mixedMsePred",
                DataType.FLOAT32
        );
        Tensor target = new Tensor(
                new float[]{0f, 1f, 2f, 3f, 4f, 5f},
                new int[]{2, 3},
                null,
                "mixedMseTarget",
                DataType.FLOAT32
        );
        Tensor diff = pred.sub(target);
        Tensor square = diff.mul(diff);
        Tensor rowMean = square.mean(1);
        Tensor loss = rowMean.sum(0, true);

        List<Tensor> graph = loss.topologicalSort();
        int diffNodeId = graph.indexOf(diff);
        int squareNodeId = graph.indexOf(square);
        int rowMeanNodeId = graph.indexOf(rowMean);
        int lossNodeId = graph.indexOf(loss);
        List<CompiledNode> nodes = CompiledNodeSnapshotter.snapshot(graph, BackendIntentPlan.empty());
        List<Integer> selectedNodeIds = List.of(diffNodeId, squareNodeId, rowMeanNodeId, lossNodeId);
        Partition partition = partition(
                "cpu-mixed-reduction-mse",
                PartitionTarget.CPU,
                selectedNodeIds,
                externalInputNodeIds(nodes, selectedNodeIds),
                List.of(GraphValueRef.node(lossNodeId)),
                List.of(GraphValueRef.node(lossNodeId))
        );

        PartitionExecutionPlan executionPlan = new PartitionExecutionPlanner().planPartition(
                partition,
                new PartitionExecutionPlanningContext(nodes, FuseConfig.inferenceDefaults())
        );

        assertTrue(executionPlan.executionUnits().stream()
                .noneMatch(unit -> unit.kind() == ExecutionUnitKind.SPECIALIZED_PRIMITIVE
                        && unit.specialization().kind() == PartitionSpecializationKind.MSE_LOSS));
        assertEquals(selectedNodeIds, executionPlan.executionUnits().stream()
                .flatMap(unit -> unit.orderedNodeIds().stream())
                .toList());
    }

    @Test
    void gpuMsePartitionRejectsSpecializationAndFallsBackToStructuralUnits() {
        Tensor pred = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{4}, null, "gpuMsePred", DataType.FLOAT32);
        Tensor target = new Tensor(new float[]{1.5f, 1f, 2.5f, 3f}, new int[]{4}, null, "gpuMseTarget", DataType.FLOAT32);
        Tensor diff = pred.sub(target);
        Tensor square = diff.mul(diff);
        Tensor loss = square.mean();

        List<CompiledNode> nodes = CompiledNodeSnapshotter.snapshot(loss.topologicalSort(), BackendIntentPlan.empty());
        int diffNodeId = nodeId(nodes, Operation.OpType.SUB);
        int squareNodeId = nodeId(nodes, Operation.OpType.MUL);
        int lossNodeId = nodeId(nodes, Operation.OpType.MEAN);
        List<Integer> selectedNodeIds = List.of(diffNodeId, squareNodeId, lossNodeId);
        Partition partition = partition(
                "gpu-mse",
                PartitionTarget.GPU_METAL,
                selectedNodeIds,
                externalInputNodeIds(nodes, selectedNodeIds),
                List.of(GraphValueRef.node(lossNodeId)),
                List.of(GraphValueRef.node(lossNodeId))
        );

        PartitionExecutionPlan executionPlan = new PartitionExecutionPlanner().planPartition(
                partition,
                new PartitionExecutionPlanningContext(nodes, FuseConfig.inferenceDefaults())
        );

        assertTrue(executionPlan.executionUnits().stream()
                .noneMatch(unit -> unit.kind() == ExecutionUnitKind.SPECIALIZED_PRIMITIVE));
        assertEquals(selectedNodeIds, executionPlan.executionUnits().stream()
                .flatMap(unit -> unit.orderedNodeIds().stream())
                .toList());
        assertTrue(executionPlan.trace().events().stream()
                .anyMatch(event -> event.contains("specialization-candidate-found:kind=MSE_LOSS")));
        assertTrue(executionPlan.trace().events().stream()
                .anyMatch(event -> event.contains("specialization-candidate-rejected:kind=MSE_LOSS")
                        && event.contains("backend-specialization-unsupported:GPU_METAL")));
    }

    @Test
    void gpuPartitionBuildsFusedElementwiseSubchainWithoutShorteningPartition() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "gpuSubchainA", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 0f, 0f, 1f, 1f, 1f}, new int[]{3, 2}, null, "gpuSubchainB", DataType.FLOAT32);
        Tensor bias = new Tensor(new float[]{0.25f, -0.5f}, new int[]{2}, null, "gpuSubchainBias", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor add = matmul.add(bias);
        Tensor relu = add.relu();
        Tensor out = relu.exp();

        List<CompiledNode> nodes = CompiledNodeSnapshotter.snapshot(out.topologicalSort(), BackendIntentPlan.empty());
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

        PartitionExecutionPlan executionPlan = new PartitionExecutionPlanner().planPartition(
                partition,
                new PartitionExecutionPlanningContext(nodes, FuseConfig.inferenceDefaults())
        );

        assertEquals(selectedNodeIds, partition.orderedNodeIds());
        assertTrue(executionPlan.executionUnits().stream().anyMatch(unit -> unit.kind() == ExecutionUnitKind.FUSED_ELEMENTWISE));
        ExecutionUnit fused = executionPlan.executionUnits().stream()
                .filter(unit -> unit.kind() == ExecutionUnitKind.FUSED_ELEMENTWISE)
                .findFirst()
                .orElseThrow();
        assertEquals(List.of(addNodeId, reluNodeId, expNodeId), fused.orderedNodeIds());
        assertTrue(fused.trace().events().stream().anyMatch(message -> message.contains("fused-subchain:")));
        assertFalse(executionPlan.executionValues().stream()
                .filter(value -> value.producerNodeId() == addNodeId || value.producerNodeId() == reluNodeId)
                .anyMatch(value -> value.transportKind() == ValueTransportKind.MATERIALIZED));
    }

    @Test
    void gpuPartitionKeepsMixedMatmulAndElementwisePartitionSelected() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "gpuMixedA", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 0f, 0f, 1f, 1f, 1f}, new int[]{3, 2}, null, "gpuMixedB", DataType.FLOAT32);
        Tensor bias = new Tensor(new float[]{0.25f, -0.5f}, new int[]{2}, null, "gpuMixedBias", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor add = matmul.add(bias);
        Tensor relu = add.relu();
        Tensor out = relu.exp();

        List<CompiledNode> nodes = CompiledNodeSnapshotter.snapshot(out.topologicalSort(), BackendIntentPlan.empty());
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

        PartitionExecutionPlan executionPlan = new PartitionExecutionPlanner().planPartition(
                partition,
                new PartitionExecutionPlanningContext(nodes, FuseConfig.inferenceDefaults())
        );

        assertEquals(selectedNodeIds, partition.orderedNodeIds());
        assertEquals(selectedNodeIds, executionPlan.executionUnits().stream()
                .flatMap(unit -> unit.orderedNodeIds().stream())
                .toList());
        assertEquals(ExecutionUnitKind.UNIT_KERNEL, executionPlan.executionUnits().getFirst().kind());
        assertEquals(ExecutionUnitKind.FUSED_ELEMENTWISE, executionPlan.executionUnits().get(1).kind());
        assertEquals(List.of(addNodeId, reluNodeId, expNodeId), executionPlan.executionUnits().get(1).orderedNodeIds());
    }

    @Test
    void gpuPartitionBuildsMatmulBiasActivationEpilogueUnit() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "gpuEpilogueA", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{1f, 0f, 0f, 1f, 1f, 1f}, new int[]{3, 2}, null, "gpuEpilogueB", DataType.FLOAT32);
        Tensor bias = new Tensor(new float[]{0.25f, -0.5f}, new int[]{2}, null, "gpuEpilogueBias", DataType.FLOAT32);
        Tensor matmul = a.matmul(b);
        Tensor add = matmul.add(bias);
        Tensor out = add.relu();

        List<CompiledNode> nodes = CompiledNodeSnapshotter.snapshot(out.topologicalSort(), BackendIntentPlan.empty());
        int matmulNodeId = nodeId(nodes, Operation.OpType.MATMUL);
        int addNodeId = nodeId(nodes, Operation.OpType.ADD);
        int reluNodeId = nodeId(nodes, Operation.OpType.RELU);
        List<Integer> selectedNodeIds = List.of(matmulNodeId, addNodeId, reluNodeId);
        Partition partition = partition(
                "gpu-epilogue-subpartition",
                PartitionTarget.GPU_METAL,
                selectedNodeIds,
                externalInputNodeIds(nodes, selectedNodeIds),
                List.of(GraphValueRef.node(reluNodeId)),
                List.of(GraphValueRef.node(reluNodeId))
        );

        PartitionExecutionPlan executionPlan = new PartitionExecutionPlanner().planPartition(
                partition,
                new PartitionExecutionPlanningContext(nodes, FuseConfig.inferenceDefaults())
        );

        assertEquals(selectedNodeIds, partition.orderedNodeIds());
        assertEquals(2, executionPlan.executionUnits().size());
        ExecutionUnit matmulUnit = executionPlan.executionUnits().getFirst();
        ExecutionUnit fusedEpilogueTail = executionPlan.executionUnits().get(1);
        assertEquals(ExecutionUnitKind.UNIT_KERNEL, matmulUnit.kind());
        assertEquals(List.of(matmulNodeId), matmulUnit.orderedNodeIds());
        assertEquals(ExecutionUnitKind.FUSED_ELEMENTWISE, fusedEpilogueTail.kind());
        assertEquals(List.of(addNodeId, reluNodeId), fusedEpilogueTail.orderedNodeIds());
        assertTrue(fusedEpilogueTail.trace().events().stream().anyMatch(message -> message.contains("fused-subchain:")));
        assertTrue(fusedEpilogueTail.trace().events().stream().anyMatch(message -> message.contains("partition-unit-node:")));
        assertFalse(fusedEpilogueTail.requiredPreparedInputNodeIds().contains(addNodeId));
        assertFalse(executionPlan.executionValues().stream()
                .filter(value -> value.producerNodeId() == addNodeId)
                .anyMatch(value -> value.transportKind() == ValueTransportKind.MATERIALIZED));
    }

    @Test
    void gpuPartitionDoesNotUseCpuFusedNodeForPartitionInternalFusion() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{4}, null, "gpuNoCpuFusedA", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{5f, 6f, 7f, 8f}, new int[]{4}, null, "gpuNoCpuFusedB", DataType.FLOAT32);
        Tensor mul = a.mul(b);
        Tensor tanh = mul.tanh();
        Tensor out = tanh.sigmoid();

        List<CompiledNode> nodes = CompiledNodeSnapshotter.snapshot(out.topologicalSort(), BackendIntentPlan.empty());
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

        PartitionExecutionPlan executionPlan = new PartitionExecutionPlanner().planPartition(
                partition,
                new PartitionExecutionPlanningContext(nodes, FuseConfig.inferenceDefaults())
        );

        assertEquals(1, executionPlan.executionUnits().size());
        ExecutionUnit unit = executionPlan.executionUnits().getFirst();
        assertEquals(ExecutionUnitKind.FUSED_ELEMENTWISE, unit.kind());
        assertTrue(unit.trace().events().stream().anyMatch(message -> message.contains("partition-unit-node:")));
        assertFalse(unit.orderedNodeIds().stream()
                .map(nodes::get)
                .map(CompiledNode::operation)
                .anyMatch(operation -> operation != null && operation.opType() == Operation.OpType.FUSED));
    }

    @Test
    void operationSemanticClassifierKeepsHighLevelOpsVisibleForPartitionPolicy() {
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

        List<CompiledNode> nodes = CompiledNodeSnapshotter.snapshot(out.topologicalSort(), BackendIntentPlan.empty());
        Partition partition = partition(
                "cpu-cont",
                PartitionTarget.CPU,
                List.of(2, 3),
                List.of(0, 1),
                List.of(GraphValueRef.node(3)),
                List.of()
        );

        PartitionExecutionPlanner planner = new PartitionExecutionPlanner();
        PartitionExecutionPlan executionPlan = planner.planPartition(partition, new PartitionExecutionPlanningContext(nodes, FuseConfig.inferenceDefaults()));

        PartitionExecutionValue output = executionPlan.executionValues().stream()
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

    private static void assertSdpaBackwardSpecializationForSingleGradient(
            SdpaBackwardOutputKind outputKind,
            int[] expectedOutputShape
    ) {
        Tensor q = new Tensor(new float[]{
                1.0f, 0.5f,
                -0.25f, 0.75f
        }, new int[]{1, 2, 2}, null, "sdpaSpecializeQ", DataType.FLOAT32);
        Tensor k = new Tensor(new float[]{
                0.25f, 1.0f,
                1.5f, -0.5f
        }, new int[]{1, 2, 2}, null, "sdpaSpecializeK", DataType.FLOAT32);
        Tensor v = new Tensor(new float[]{
                2.0f, -1.0f,
                0.5f, 3.0f
        }, new int[]{1, 2, 2}, null, "sdpaSpecializeV", DataType.FLOAT32);
        switch (outputKind) {
            case QUERY -> q.setRequiresGrad(true);
            case KEY -> k.setRequiresGrad(true);
            case VALUE -> v.setRequiresGrad(true);
        }
        Tensor loss = q.scaledDotProductAttention(k, v, AttentionOptions.defaults().withScale(0.5d)).sum();
        CompiledGraph compiled = CompiledGraph.compile(loss, CompileConfig.noGraphOptimizationBaseline());
        List<CompiledNode> nodes = compiled.program().compiledNodes();
        int terminalNodeId = backwardMatmulTargetByShape(nodes, expectedOutputShape);
        List<Integer> selectedNodeIds = backwardOpClosure(nodes, terminalNodeId);
        Partition partition = partition(
                "cpu-sdpa-backward-" + outputKind.name().toLowerCase(java.util.Locale.ROOT),
                PartitionTarget.CPU,
                selectedNodeIds,
                externalInputNodeIds(nodes, selectedNodeIds),
                List.of(GraphValueRef.node(terminalNodeId)),
                List.of(GraphValueRef.node(terminalNodeId))
        );

        PartitionExecutionPlan executionPlan = new PartitionExecutionPlanner().planPartition(
                partition,
                new PartitionExecutionPlanningContext(nodes, FuseConfig.trainingDefaults())
        );

        assertEquals(1, executionPlan.executionUnits().size());
        ExecutionUnit unit = executionPlan.executionUnits().getFirst();
        assertEquals(ExecutionUnitKind.SPECIALIZED_PRIMITIVE, unit.kind());
        assertEquals(PartitionSpecializationKind.SDPA_BACKWARD, unit.specialization().kind());
        assertEquals(selectedNodeIds, unit.orderedNodeIds());
        SdpaBackwardSpecializationPayload payload =
                (SdpaBackwardSpecializationPayload) unit.specialization().payload();
        assertEquals(expectedSdpaBackwardInputRefs(payload), unit.inputValueRefs());
        assertEquals(outputKind, payload.outputKind());
        assertEquals(0.5d, payload.scale(), 0.0d);
        assertFalse(payload.hasMask());
        assertTrue(payload.weightsNodeId() >= 0);
        assertTrue(payload.outGradNodeId() >= 0);
        if (outputKind == SdpaBackwardOutputKind.QUERY) {
            assertTrue(payload.keyNodeId() >= 0);
            assertTrue(payload.valueNodeId() >= 0);
        } else if (outputKind == SdpaBackwardOutputKind.KEY) {
            assertTrue(payload.queryNodeId() >= 0);
            assertTrue(payload.valueNodeId() >= 0);
        }
        assertTrue(executionPlan.trace().events().stream()
                .anyMatch(event -> event.contains("specialization-candidate-accepted:kind=SDPA_BACKWARD")));
        assertFalse(nodes.stream()
                .map(CompiledNode::operation)
                .filter(java.util.Objects::nonNull)
                .map(Operation::opType)
                .anyMatch(opType -> opType == Operation.OpType.SCALED_DOT_PRODUCT_ATTENTION_BACKWARD));
    }

    private static int backwardMatmulTargetByShape(List<CompiledNode> nodes, int[] shape) {
        return nodes.stream()
                .filter(CompiledNode::backwardNode)
                .filter(CompiledNode::gradientTarget)
                .filter(node -> node.operation() != null && node.operation().opType() == Operation.OpType.MATMUL)
                .filter(node -> java.util.Arrays.equals(node.shape(), shape))
                .map(CompiledNode::id)
                .reduce((first, second) -> second)
                .orElseThrow();
    }

    private static List<Integer> backwardOpClosure(List<CompiledNode> nodes, int terminalNodeId) {
        java.util.LinkedHashSet<Integer> out = new java.util.LinkedHashSet<>();
        collectBackwardOpClosure(nodes, terminalNodeId, terminalNodeId, out);
        return nodes.stream()
                .map(CompiledNode::id)
                .filter(out::contains)
                .toList();
    }

    private static List<GraphValueRef> expectedSdpaBackwardInputRefs(SdpaBackwardSpecializationPayload payload) {
        java.util.LinkedHashSet<GraphValueRef> out = new java.util.LinkedHashSet<>();
        out.add(GraphValueRef.node(payload.weightsNodeId()));
        out.add(GraphValueRef.node(payload.outGradNodeId()));
        switch (payload.outputKind()) {
            case QUERY -> {
                out.add(GraphValueRef.node(payload.keyNodeId()));
                out.add(GraphValueRef.node(payload.valueNodeId()));
            }
            case KEY -> {
                out.add(GraphValueRef.node(payload.queryNodeId()));
                out.add(GraphValueRef.node(payload.valueNodeId()));
            }
            case VALUE -> {
            }
        }
        if (payload.hasMask() && payload.outputKind() != SdpaBackwardOutputKind.VALUE) {
            out.add(GraphValueRef.node(payload.maskNodeId()));
        }
        return List.copyOf(out);
    }

    private static void collectBackwardOpClosure(
            List<CompiledNode> nodes,
            int nodeId,
            int terminalNodeId,
            java.util.Set<Integer> out
    ) {
        CompiledNode node = nodes.get(nodeId);
        if (node.operation() == null || !node.backwardNode()) {
            return;
        }
        if (node.id() != terminalNodeId && node.gradientTarget()) {
            return;
        }
        if (!out.add(nodeId)) {
            return;
        }
        for (int inputId : node.inputIds()) {
            collectBackwardOpClosure(nodes, inputId, terminalNodeId, out);
        }
    }
}
