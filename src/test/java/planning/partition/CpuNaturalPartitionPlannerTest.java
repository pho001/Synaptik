package planning.partition;

import planning.descriptor.CompiledTensorDescriptorBuilder;
import planning.descriptor.CompiledTensorDescriptorIndex;
import planning.intent.BackendIntentPlan;

import backend.cpu.partition.CpuBackendPartitionCapability;
import config.optimizer.CpuPartitionConfig;
import config.compile.CompileConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import graph.compile.CompiledNodeSnapshotter;
import graph.model.CompiledNode;
import planning.value.GraphValueRef;
import planning.partition.cost.AcceleratorPartitionScoreModel;
import planning.partition.execution.PartitionExecutionPlanner;
import planning.partition.execution.ExecutionUnitKind;
import planning.partition.execution.PartitionExecutionPlan;
import planning.partition.execution.PartitionExecutionPlanningContext;
import planning.partition.specialization.PartitionSpecializationKind;
import operations.Operation;
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class CpuNaturalPartitionPlannerTest {
    @Test
    void createsCpuExecutionPartitionContainingElementwiseAndUnitBoundary() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{4}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{5f, 6f, 7f, 8f}, new int[]{4}, null, "b", DataType.FLOAT32);
        Tensor sum = a.add(b).relu().sum();

        List<CompiledNode> nodes = CompiledNodeSnapshotter.snapshot(sum.topologicalSort(), BackendIntentPlan.empty());
        PartitionPlanningContext context = new PartitionPlanningContext(
                false,
                nodes,
                CompiledTensorDescriptorBuilder.build(nodes),
                consumers(nodes)
        );

        PartitionPlanningResult result = new CpuNaturalPartitionPlanner().plan(new PartitionPlanningRequest(
                PartitionPlannerStrategy.CPU_NATURAL_EXECUTION_PARTITION,
                PartitionTarget.CPU,
                context,
                new AcceleratorPartitionScoreModel.PlannerPolicy(64, 512, 1, 1, 1, 1, 1, 1),
                new CpuBackendPartitionCapability(),
                Set.of(GraphValueRef.node(nodes.size() - 1))
        ));

        assertEquals(1, result.partitions().size());
        Partition partition = result.partitions().getFirst();
        assertEquals(PartitionKind.CPU_EXECUTION, partition.partitionKind());
        assertEquals(PartitionPlannerStrategy.CPU_NATURAL_EXECUTION_PARTITION, partition.plannerStrategy());
        assertTrue(partition.orderedNodeIds().size() >= 3);

        PartitionExecutionPlan executionPlan = new PartitionExecutionPlanner().planPartition(
                partition,
                new PartitionExecutionPlanningContext(nodes, config.optimizer.FuseConfig.inferenceDefaults())
        );

        assertTrue(executionPlan.executionUnits().stream().anyMatch(unit -> unit.kind() == ExecutionUnitKind.FUSED_ELEMENTWISE));
        assertTrue(executionPlan.executionUnits().stream().anyMatch(unit -> unit.kind() == ExecutionUnitKind.UNIT_KERNEL));
    }

    @Test
    void partitionIntentRuleUsesCpuNaturalStrategyForDefaultCpuGraph() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{4}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{5f, 6f, 7f, 8f}, new int[]{4}, null, "b", DataType.FLOAT32);
        Tensor root = a.add(b).relu().sum();

        CompiledGraph result = CompiledGraph.compile(root, CompileConfig.inference());

        assertEquals(1, result.program().partitions().size());
        Partition partition = result.program().partitions().getFirst();
        assertEquals(PartitionKind.CPU_EXECUTION, partition.partitionKind());
        assertEquals(PartitionPlannerStrategy.CPU_NATURAL_EXECUTION_PARTITION, partition.plannerStrategy());
        assertTrue(partition.orderedNodeIds().size() >= 3);
    }

    @Test
    void compiledGraphSpecializesNestedMeanMsePartition() {
        Tensor prediction = new Tensor(
                new float[]{1f, 2f, 3f, 4f, 5f, 6f},
                new int[]{2, 3},
                null,
                "plannerNestedMsePrediction",
                DataType.FLOAT32
        );
        Tensor target = new Tensor(
                new float[]{0f, 1f, 2f, 3f, 4f, 5f},
                new int[]{2, 3},
                null,
                "plannerNestedMseTarget",
                DataType.FLOAT32
        );
        Tensor diff = prediction.sub(target);
        Tensor root = diff.mul(diff).mean(1).mean(0, true);

        CompiledGraph compiled = CompiledGraph.compile(root, CompileConfig.inference());

        assertTrue(compiled.program().executablePartitions().stream()
                .flatMap(executablePartition -> executablePartition.executionPlan().executionUnits().stream())
                .anyMatch(unit -> unit.kind() == ExecutionUnitKind.SPECIALIZED_PRIMITIVE
                        && unit.specialization().kind() == PartitionSpecializationKind.MSE_LOSS
                        && unit.orderedNodeIds().size() == 4));
    }

    @Test
    void compiledGraphSnapshotHonorsCpuPartitionPolicy() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{4}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{5f, 6f, 7f, 8f}, new int[]{4}, null, "b", DataType.FLOAT32);
        Tensor root = a.add(b).relu().sum();

        CompiledGraph enabled = CompiledGraph.compile(root, CompileConfig.inference());

        assertEquals(1, enabled.program().partitions().size());
        assertEquals(
                PartitionPlannerStrategy.CPU_NATURAL_EXECUTION_PARTITION,
                enabled.program().partitions().getFirst().plannerStrategy()
        );

        CompiledGraph disabled = CompiledGraph.compile(
                root,
                CompileConfig.inference().withBackendPlanning(
                        CompileConfig.inference().backendPlanning().withCpuPartitions(CpuPartitionConfig.off())
                )
        );

        assertTrue(disabled.program().partitions().isEmpty());
    }

    @Test
    void cpuNaturalPartitionsRemainAvailableWhenAcceleratorBenefitIsAmbiguous() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{4}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{5f, 6f, 7f, 8f}, new int[]{4}, null, "b", DataType.FLOAT32);
        Tensor root = a.add(b).relu().sum();

        CompiledGraph compiled = CompiledGraph.compile(
                root,
                CompileConfig.inference().withBackendPlanning(config.compile.BackendPlanningConfig.autoAccelerator().withOwnershipPlanner(config.compile.PartitionOwnershipPlannerStrategy.SCORED))
        );

        assertTrue(compiled.program().partitions().stream()
                .anyMatch(partition -> partition.plannerStrategy()
                        == PartitionPlannerStrategy.CPU_NATURAL_EXECUTION_PARTITION));
    }

    @Test
    void elementwiseIslandPolicyDoesNotAbsorbUnitKernelBoundaries() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{7f, 8f, 9f, 10f, 11f, 12f}, new int[]{3, 2}, null, "b", DataType.FLOAT32);
        Tensor root = a.matmul(b).relu();

        CompiledGraph result = CompiledGraph.compile(
                root,
                CompileConfig.inference().withBackendPlanning(
                        CompileConfig.inference().backendPlanning().withCpuPartitions(CpuPartitionConfig.elementwiseIslands())
                )
        );

        assertEquals(1, result.program().partitions().size());
        Partition partition = result.program().partitions().getFirst();
        assertEquals(List.of(3), partition.orderedNodeIds());
    }

    @Test
    void compiledGraphCreatesExactMatmulReluPartitionBeforeTrailingCpuConsumer() {
        Tensor a = new Tensor(new float[]{1f, -2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{7f, 8f, -9f, 10f, 11f, -12f}, new int[]{3, 2}, null, "b", DataType.FLOAT32);
        Tensor root = a.matmul(b).relu().sum();

        CompiledGraph result = CompiledGraph.compile(root, CompileConfig.inference());

        assertTrue(result.program().partitions().stream()
                .map(partition -> partitionOps(result, partition))
                .anyMatch(ops -> ops.equals(List.of(Operation.OpType.MATMUL, Operation.OpType.RELU))));
    }

    @Test
    void compiledGraphCreatesExactLinearReluPartitionBeforeTrailingCpuConsumer() {
        Tensor a = new Tensor(new float[]{1f, -2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{7f, 8f, -9f, 10f, 11f, -12f}, new int[]{3, 2}, null, "b", DataType.FLOAT32);
        Tensor bias = new Tensor(new float[]{1f, -4f}, new int[]{2}, null, "bias", DataType.FLOAT32);
        Tensor root = a.matmul(b).add(bias).relu().sum();

        CompiledGraph result = CompiledGraph.compile(root, CompileConfig.inference());

        assertTrue(result.program().partitions().stream()
                .map(partition -> partitionOps(result, partition))
                .anyMatch(ops -> ops.equals(List.of(Operation.OpType.LINEAR, Operation.OpType.RELU))));
    }

    @Test
    void compiledGraphCreatesExactLinearBiasPartitionBeforeTrailingCpuConsumer() {
        Tensor a = new Tensor(new float[]{1f, -2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "linearBiasA", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{7f, 8f, -9f, 10f, 11f, -12f}, new int[]{3, 2}, null, "linearBiasB", DataType.FLOAT32);
        Tensor bias = new Tensor(new float[]{1f, -4f}, new int[]{2}, null, "linearBias", DataType.FLOAT32);
        Tensor root = a.linear(b, bias).sum();

        CompiledGraph result = CompiledGraph.compile(root, CompileConfig.inference());

        assertTrue(result.program().partitions().stream()
                .map(partition -> partitionOps(result, partition))
                .anyMatch(ops -> ops.equals(List.of(Operation.OpType.LINEAR))));
    }

    @Test
    void compiledGraphCreatesExactMatmulAddBiasPartitionBeforeTrailingCpuConsumer() {
        Tensor a = new Tensor(new float[]{1f, -2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "matmulBiasA", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{7f, 8f, -9f, 10f, 11f, -12f}, new int[]{3, 2}, null, "matmulBiasB", DataType.FLOAT32);
        Tensor rowBias = new Tensor(new float[]{1f, -4f}, new int[]{1, 2}, null, "matmulRowBias", DataType.FLOAT32);
        Tensor root = a.matmul(b).add(rowBias).sum();

        CompiledGraph result = CompiledGraph.compile(root, CompileConfig.inference());

        assertTrue(result.program().partitions().stream()
                .map(partition -> partitionOps(result, partition))
                .anyMatch(ops -> ops.equals(List.of(Operation.OpType.MATMUL, Operation.OpType.ADD))));
    }

    private static List<Operation.OpType> partitionOps(CompiledGraph graph, Partition partition) {
        return partition.orderedNodeIds().stream()
                .map(nodeId -> graph.program().compiledNodes().get(nodeId))
                .map(CompiledNode::operation)
                .map(operation -> operation == null ? Operation.OpType.UNKNOWN : operation.opType())
                .toList();
    }

    private static Map<Integer, List<CompiledNode>> consumers(List<CompiledNode> graph) {
        Map<Integer, List<CompiledNode>> consumers = new HashMap<>();
        for (CompiledNode node : graph) {
            consumers.computeIfAbsent(node.id(), ignored -> new ArrayList<>());
        }
        for (CompiledNode node : graph) {
            for (int inputId : node.inputIds()) {
                consumers.computeIfAbsent(inputId, ignored -> new ArrayList<>()).add(node);
            }
        }
        return consumers;
    }
}
