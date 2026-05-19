package graph.optimizer.region;

import backend.runtime.ExecutionMode;
import config.optimizer.FuseConfig;
import graph.execution.trace.PartitionDecisionTrace;
import graph.optimizer.partition.Partition;
import graph.optimizer.partition.PartitionBoundaryReason;
import graph.optimizer.partition.PartitionEdge;
import graph.optimizer.partition.PartitionPlannerStrategy;
import graph.optimizer.partition.PartitionTarget;
import graph.optimizer.partition.PartitionValue;
import graph.optimizer.GraphValueRef;
import graph.optimizer.state.OptimizerState;
import org.junit.jupiter.api.Test;
import operations.Operation;
import tensor.DataType;
import tensor.Tensor;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

class RegionOptimizationRuleTest {
    @Test
    void regionOptimizationRuleUsesDefaultRegionOptimizerWhenPartitionsArePresent() {
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
                new graph.optimizer.partition.cost.AcceleratorPartitionScoreModel.CandidateMetrics(2, 1, 2, 0, 1),
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

        OptimizerState initial = OptimizerState.ofGraph(out.topologicalSort(), out)
                .withExecutionMetadata(ExecutionMode.FORWARD, false, out.topologicalSort().indexOf(out))
                .withPartitions(List.of(partition));

        OptimizerState result = new RegionOptimizationRule(FuseConfig.inferenceDefaults()).apply(initial);

        assertFalse(result.optimizedRegions().isEmpty());
        assertEquals(1, result.optimizedRegions().size());
        assertEquals(1, result.optimizedRegions().getFirst().executionUnits().size());
        assertEquals(initial.graph(), result.graph());
    }

    @Test
    void regionOptimizationRuleOptimizesNonCpuPartitionsToo() {
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
                new graph.optimizer.partition.cost.AcceleratorPartitionScoreModel.CandidateMetrics(2, 1, 2, 0, 1),
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

        OptimizerState initial = OptimizerState.ofGraph(out.topologicalSort(), out)
                .withExecutionMetadata(ExecutionMode.FORWARD, false, out.topologicalSort().indexOf(out))
                .withPartitions(List.of(partition));

        OptimizerState result = new RegionOptimizationRule(FuseConfig.inferenceDefaults()).apply(initial);

        assertEquals(1, result.optimizedRegions().size());
        assertEquals(PartitionTarget.GPU_METAL, result.optimizedRegions().getFirst().target());
        assertFalse(result.optimizedRegions().getFirst().executionUnits().isEmpty());
    }

    @Test
    void regionOptimizationRuleDoesNotMutateGraphWhenPartitionStateIsAbsent() {
        Tensor a = new Tensor(new float[]{1f, 2f, 3f, 4f}, new int[]{4}, null, "a", DataType.FLOAT32);
        Tensor b = new Tensor(new float[]{5f, 6f, 7f, 8f}, new int[]{4}, null, "b", DataType.FLOAT32);
        Tensor add = a.add(b);
        Tensor out = add.relu();

        OptimizerState initial = OptimizerState.ofGraph(out.topologicalSort(), out)
                .withExecutionMetadata(ExecutionMode.FORWARD, false, out.topologicalSort().indexOf(out));

        OptimizerState result = new RegionOptimizationRule(FuseConfig.inferenceDefaults()).apply(initial);

        assertEquals(initial.graph(), result.graph());
        assertEquals(0, result.optimizedRegions().size());
        assertEquals(Operation.OpType.ADD, add.getOperation().opType());
        assertEquals(Operation.OpType.RELU, out.getOperation().opType());
    }
}
