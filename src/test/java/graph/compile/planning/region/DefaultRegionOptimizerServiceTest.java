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
import org.junit.jupiter.api.Test;
import tensor.DataType;
import tensor.Tensor;
import graph.compile.intent.BackendIntentPlan;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

class DefaultRegionOptimizerServiceTest {
    @Test
    void defaultRegionOptimizerBuildsCpuExecutionUnits() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{4}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{5f, 6f, 7f, 8f}, new int[]{4}, null, "b", DataType.FLOAT32);
        Tensor add = a.add(b);
        Tensor out = add.relu();

        Partition partition = new Partition(
                "cpu-partition",
                PartitionTarget.CPU,
                List.of(2, 3),
                List.of(
                        new PartitionValue(GraphValueRef.node(2), 2),
                        new PartitionValue(GraphValueRef.node(3), 3)
                ),
                List.of(new PartitionEdge(2, 3)),
                List.of(0, 1),
                List.of(GraphValueRef.node(3)),
                2,
                List.of(GraphValueRef.node(3)),
                List.of(),
                List.of(PartitionBoundaryReason.NONE),
                2L,
                new graph.compile.planning.partition.cost.AcceleratorPartitionScoreModel.CandidateMetrics(2, 1, 2, 0, 1),
                PartitionPlannerStrategy.GREEDY_MAX_REGION,
                new PartitionDecisionTrace(
                        PartitionPlannerStrategy.GREEDY_MAX_REGION,
                        PartitionTarget.CPU,
                        2,
                        true,
                        "test",
                        List.of(2, 3),
                        List.of(2, 3),
                        List.of("ADD", "RELU"),
                        2L,
                        0.0d,
                        0.0d,
                        0,
                        false,
                        -1
                )
        );

        List<Tensor> graph = out.topologicalSort();
        OptimizedRegion result = new DefaultRegionOptimizer().optimize(
                partition,
                new RegionOptimizationContext(CompiledNode.snapshot(graph, BackendIntentPlan.empty()), FuseConfig.inferenceDefaults())
        );

        assertEquals(1, result.executionUnits().size());
    }

    @Test
    void defaultRegionOptimizerOptimizesNonCpuPartitionsToo() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{4}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{5f, 6f, 7f, 8f}, new int[]{4}, null, "b", DataType.FLOAT32);
        Tensor add = a.add(b);
        Tensor out = add.relu();

        Partition partition = new Partition(
                "gpu-partition",
                PartitionTarget.GPU_METAL,
                List.of(2, 3),
                List.of(
                        new PartitionValue(GraphValueRef.node(2), 2),
                        new PartitionValue(GraphValueRef.node(3), 3)
                ),
                List.of(new PartitionEdge(2, 3)),
                List.of(0, 1),
                List.of(GraphValueRef.node(3)),
                2,
                List.of(GraphValueRef.node(3)),
                List.of(),
                List.of(PartitionBoundaryReason.NONE),
                2L,
                new graph.compile.planning.partition.cost.AcceleratorPartitionScoreModel.CandidateMetrics(2, 1, 2, 0, 1),
                PartitionPlannerStrategy.GREEDY_MAX_REGION,
                new PartitionDecisionTrace(
                        PartitionPlannerStrategy.GREEDY_MAX_REGION,
                        PartitionTarget.GPU_METAL,
                        2,
                        true,
                        "test",
                        List.of(2, 3),
                        List.of(2, 3),
                        List.of("ADD", "RELU"),
                        2L,
                        0.0d,
                        0.0d,
                        0,
                        false,
                        -1
                )
        );

        List<Tensor> graph = out.topologicalSort();
        OptimizedRegion result = new DefaultRegionOptimizer().optimize(
                partition,
                new RegionOptimizationContext(CompiledNode.snapshot(graph, BackendIntentPlan.empty()), FuseConfig.inferenceDefaults())
        );

        assertEquals(PartitionTarget.GPU_METAL, result.target());
        assertFalse(result.executionUnits().isEmpty());
    }
}
