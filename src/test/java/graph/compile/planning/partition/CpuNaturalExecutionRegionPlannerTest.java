package graph.compile.planning.partition;

import graph.compile.descriptor.CompiledTensorDescriptorBuilder;
import graph.compile.descriptor.CompiledTensorDescriptorIndex;

import backend.cpu.partition.CpuRegionLegalityAdapter;
import config.optimizer.CpuRegionConfig;
import config.compile.CompileConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import graph.CompiledNode;
import graph.compile.planning.value.GraphValueRef;
import graph.compile.planning.partition.cost.AcceleratorPartitionScoreModel;
import graph.compile.planning.region.DefaultRegionOptimizer;
import graph.compile.planning.region.ExecutionUnitKind;
import graph.compile.planning.region.OptimizedRegion;
import graph.compile.planning.region.RegionOptimizationContext;
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

class CpuNaturalExecutionRegionPlannerTest {
    @Test
    void createsCpuExecutionRegionContainingElementwiseAndUnitBoundary() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{4}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{5f, 6f, 7f, 8f}, new int[]{4}, null, "b", DataType.FLOAT32);
        Tensor sum = a.add(b).relu().sum();

        List<CompiledNode> nodes = CompiledNode.snapshot(sum.topologicalSort());
        PartitionPlanningContext context = new PartitionPlanningContext(
                false,
                nodes,
                CompiledTensorDescriptorBuilder.build(nodes),
                consumers(nodes)
        );

        PartitionPlanningResult result = new CpuNaturalExecutionRegionPlanner().plan(new PartitionPlanningRequest(
                PartitionPlannerStrategy.CPU_NATURAL_EXECUTION_REGION,
                PartitionTarget.CPU,
                context,
                new AcceleratorPartitionScoreModel.PlannerPolicy(64, 512, 1, 1, 1, 1, 1, 1),
                new CpuRegionLegalityAdapter(),
                Set.of(GraphValueRef.node(nodes.size() - 1))
        ));

        assertEquals(1, result.partitions().size());
        Partition partition = result.partitions().getFirst();
        assertEquals(ExecutionRegionKind.CPU_EXECUTION, partition.regionKind());
        assertEquals(PartitionPlannerStrategy.CPU_NATURAL_EXECUTION_REGION, partition.plannerStrategy());
        assertTrue(partition.orderedNodeIds().size() >= 3);

        OptimizedRegion region = new DefaultRegionOptimizer().optimize(
                partition,
                new RegionOptimizationContext(nodes, config.optimizer.FuseConfig.inferenceDefaults())
        );

        assertTrue(region.executionUnits().stream().anyMatch(unit -> unit.kind() == ExecutionUnitKind.FUSED_ELEMENTWISE));
        assertTrue(region.executionUnits().stream().anyMatch(unit -> unit.kind() == ExecutionUnitKind.UNIT_KERNEL));
    }

    @Test
    void partitionIntentRuleUsesCpuNaturalStrategyForDefaultCpuGraph() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{4}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{5f, 6f, 7f, 8f}, new int[]{4}, null, "b", DataType.FLOAT32);
        Tensor root = a.add(b).relu().sum();

        CompiledGraph result = CompiledGraph.compile(root, CompileConfig.inference());

        assertEquals(1, result.compileArtifacts().partitions().size());
        Partition partition = result.compileArtifacts().partitions().getFirst();
        assertEquals(ExecutionRegionKind.CPU_EXECUTION, partition.regionKind());
        assertEquals(PartitionPlannerStrategy.CPU_NATURAL_EXECUTION_REGION, partition.plannerStrategy());
        assertTrue(partition.orderedNodeIds().size() >= 3);
    }

    @Test
    void compiledGraphSnapshotHonorsCpuRegionPolicy() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{4}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{5f, 6f, 7f, 8f}, new int[]{4}, null, "b", DataType.FLOAT32);
        Tensor root = a.add(b).relu().sum();

        CompiledGraph enabled = CompiledGraph.compile(root, CompileConfig.inference());

        assertEquals(1, enabled.compileArtifacts().partitions().size());
        assertEquals(
                PartitionPlannerStrategy.CPU_NATURAL_EXECUTION_REGION,
                enabled.compileArtifacts().partitions().getFirst().plannerStrategy()
        );

        CompiledGraph disabled = CompiledGraph.compile(
                root,
                CompileConfig.inference().withBackendPlanning(
                        CompileConfig.inference().backendPlanning().withCpuRegions(CpuRegionConfig.off())
                )
        );

        assertTrue(disabled.compileArtifacts().partitions().isEmpty());
    }

    @Test
    void cpuNaturalRegionsRemainAvailableWhenAcceleratorBenefitIsAmbiguous() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{4}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{5f, 6f, 7f, 8f}, new int[]{4}, null, "b", DataType.FLOAT32);
        Tensor root = a.add(b).relu().sum();

        CompiledGraph compiled = CompiledGraph.compile(
                root,
                CompileConfig.inference().withBackendPlanning(config.compile.BackendPlanningConfig.autoAccelerator().withOwnershipPlanner(config.compile.RegionOwnershipPlannerStrategy.SCORED))
        );

        assertTrue(compiled.compileArtifacts().partitions().stream()
                .anyMatch(partition -> partition.plannerStrategy()
                        == PartitionPlannerStrategy.CPU_NATURAL_EXECUTION_REGION));
    }

    @Test
    void elementwiseIslandPolicyDoesNotAbsorbUnitKernelBoundaries() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{7f, 8f, 9f, 10f, 11f, 12f}, new int[]{3, 2}, null, "b", DataType.FLOAT32);
        Tensor root = a.matmul(b).relu();

        CompiledGraph result = CompiledGraph.compile(
                root,
                CompileConfig.inference().withBackendPlanning(
                        CompileConfig.inference().backendPlanning().withCpuRegions(CpuRegionConfig.elementwiseIslands())
                )
        );

        assertEquals(1, result.compileArtifacts().partitions().size());
        Partition partition = result.compileArtifacts().partitions().getFirst();
        assertEquals(List.of(3), partition.orderedNodeIds());
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
