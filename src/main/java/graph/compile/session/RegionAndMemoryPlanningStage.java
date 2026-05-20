package graph.compile.session;

import backend.runtime.ExecutionMode;
import config.compile.CompileConfig;
import graph.CompiledNode;
import graph.compile.planning.memory.MemoryPlan;
import graph.compile.planning.memory.MemoryPlanner;
import graph.compile.planning.memory.MemoryPlannerPolicy;
import graph.compile.planning.memory.MemoryPlanningInput;
import graph.compile.planning.partition.PartitionPlan;
import graph.compile.planning.partition.PlannedPartition;
import graph.compile.planning.region.DefaultRegionOptimizer;
import graph.compile.planning.region.OptimizedRegion;
import graph.compile.planning.region.RegionOptimizationContext;
import graph.optimizer.state.OptimizerState;
import tensor.Tensor;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Finalizes region optimization, optimizer execution metadata, and runtime memory planning.
 */
final class RegionAndMemoryPlanningStage {
    private RegionAndMemoryPlanningStage() {
    }

    record Result(
            List<OptimizedRegion> optimizedRegions,
            OptimizerState optimizerState,
            MemoryPlan memoryPlan,
            Map<String, PartitionPlan> partitionPlansById
    ) {
        public Result {
            optimizedRegions = List.copyOf(optimizedRegions == null ? List.of() : optimizedRegions);
            optimizerState = Objects.requireNonNull(optimizerState, "optimizerState cannot be null");
            partitionPlansById = Map.copyOf(partitionPlansById == null ? Map.of() : partitionPlansById);
        }
    }

    static Result plan(
            CompileConfig compileConfig,
            List<PlannedPartition> plannedPartitions,
            List<CompiledNode> compiledNodes,
            OptimizerState optimizerState,
            List<Tensor> graph,
            Tensor forwardOutput,
            boolean supportsBackward,
            int forwardBoundaryNodeId
    ) {
        CompileConfig config = Objects.requireNonNull(compileConfig, "compileConfig cannot be null");
        List<PlannedPartition> partitions = List.copyOf(plannedPartitions == null ? List.of() : plannedPartitions);
        List<CompiledNode> nodes = List.copyOf(compiledNodes == null ? List.of() : compiledNodes);
        List<OptimizedRegion> optimizedRegions = optimizedRegions(config, partitions, nodes);
        OptimizerState base = optimizerState == null
                ? OptimizerState.ofGraph(graph, forwardOutput)
                : optimizerState;
        ExecutionMode executionMode = supportsBackward ? ExecutionMode.FORWARD_BACKWARD : ExecutionMode.FORWARD;
        OptimizerState withMetadata = base.withExecutionMetadata(
                executionMode,
                supportsBackward,
                forwardBoundaryNodeId
        );
        Map<String, PartitionPlan> partitionPlansById = planByPartitionId(partitions);
        boolean memoryRequired = !partitions.isEmpty();
        MemoryPlan memoryPlan = (config.memoryPlanning().enabled() || memoryRequired)
                ? MemoryPlanner.plan(
                        new MemoryPlanningInput(
                                nodes,
                                optimizedRegions,
                                partitionPlansById,
                                executionMode,
                                supportsBackward,
                                forwardBoundaryNodeId
                        ),
                        MemoryPlannerPolicy.fromConfig(config.memoryPlanning().memory())
                )
                : null;
        return new Result(optimizedRegions, withMetadata, memoryPlan, partitionPlansById);
    }

    private static List<OptimizedRegion> optimizedRegions(
            CompileConfig compileConfig,
            List<PlannedPartition> plannedPartitions,
            List<CompiledNode> compiledNodes
    ) {
        if (!compileConfig.regionOptimization().enabled()) {
            return List.of();
        }
        DefaultRegionOptimizer optimizer = new DefaultRegionOptimizer();
        RegionOptimizationContext context = new RegionOptimizationContext(
                compiledNodes,
                compileConfig.regionOptimization().fuse(),
                compileConfig.regionOptimization().cpuFusion()
        );
        return plannedPartitions.stream()
                .map(PlannedPartition::partition)
                .map(partition -> optimizer.optimize(partition, context))
                .toList();
    }

    private static Map<String, PartitionPlan> planByPartitionId(List<PlannedPartition> plannedPartitions) {
        HashMap<String, PartitionPlan> out = new HashMap<>();
        for (PlannedPartition plannedPartition : plannedPartitions) {
            if (plannedPartition == null || plannedPartition.partition() == null || plannedPartition.plan() == null) {
                continue;
            }
            out.put(plannedPartition.partition().partitionId(), plannedPartition.plan());
        }
        return out;
    }
}
