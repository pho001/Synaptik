package graph.optimizer.partition;

import backend.cpu.partition.CpuRegionLegalityAdapter;
import config.optimizer.CpuRegionConfig;
import config.optimizer.OffloadConfig;
import config.optimizer.OptimizerConfig;
import config.optimizer.PartitionConfig;
import config.runtime.RuntimeConfig;
import graph.CompiledGraph;
import graph.CompiledNode;
import graph.optimizer.partition.cost.AcceleratorPartitionScoreModel;
import graph.optimizer.state.OptimizerState;
import graph.optimizer.region.DefaultRegionOptimizer;
import graph.optimizer.region.ExecutionUnitKind;
import graph.optimizer.region.OptimizedRegion;
import graph.optimizer.region.RegionOptimizationContext;
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
                RuntimeConfig.inferenceDefaults(),
                false,
                nodes,
                consumers(nodes)
        );

        PartitionPlanningResult result = new CpuNaturalExecutionRegionPlanner().plan(new PartitionPlanningRequest(
                PartitionPlannerStrategy.CPU_NATURAL_EXECUTION_REGION,
                PartitionTarget.CPU,
                context,
                new AcceleratorPartitionScoreModel.PlannerPolicy(64, 512, 1, 1, 1, 1, 1, 1),
                new CpuRegionLegalityAdapter(),
                Set.of(PartitionValueRef.ofNode(nodes.size() - 1))
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

        OptimizerState result = new PartitionIntentRule().apply(OptimizerState.ofGraph(root.topologicalSort(), root));

        assertEquals(1, result.partitions().size());
        Partition partition = result.partitions().getFirst();
        assertEquals(ExecutionRegionKind.CPU_EXECUTION, partition.regionKind());
        assertEquals(PartitionPlannerStrategy.CPU_NATURAL_EXECUTION_REGION, partition.plannerStrategy());
        assertTrue(partition.orderedNodeIds().size() >= 3);
    }

    @Test
    void compiledGraphSnapshotHonorsCpuRegionPolicy() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{4}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{5f, 6f, 7f, 8f}, new int[]{4}, null, "b", DataType.FLOAT32);
        Tensor root = a.add(b).relu().sum();

        CompiledGraph enabled = CompiledGraph.compile(root, OptimizerConfig.inferenceDefaults());

        assertEquals(1, enabled.compileArtifacts().partitions().size());
        assertEquals(
                PartitionPlannerStrategy.CPU_NATURAL_EXECUTION_REGION,
                enabled.compileArtifacts().partitions().getFirst().plannerStrategy()
        );

        CompiledGraph disabled = CompiledGraph.compile(
                root,
                OptimizerConfig.inferenceDefaults().withCpuRegion(CpuRegionConfig.off())
        );

        assertTrue(disabled.compileArtifacts().partitions().isEmpty());
    }

    @Test
    void elementwiseIslandPolicyDoesNotAbsorbUnitKernelBoundaries() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f, 5f, 6f}, new int[]{2, 3}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{7f, 8f, 9f, 10f, 11f, 12f}, new int[]{3, 2}, null, "b", DataType.FLOAT32);
        Tensor root = a.matmul(b).relu();

        OptimizerState result = new PartitionIntentRule(
                PartitionConfig.defaults(),
                OffloadConfig.defaults(),
                CpuRegionConfig.elementwiseIslands()
        ).apply(OptimizerState.ofGraph(root.topologicalSort(), root));

        assertEquals(1, result.partitions().size());
        Partition partition = result.partitions().getFirst();
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
