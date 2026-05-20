package graph.compile.planning;

import config.optimizer.CpuRegionConfig;
import graph.compile.planning.partition.PartitionPlannerStrategy;
import graph.compile.planning.partition.PartitionSourcePolicy;
import graph.compile.planning.partition.PartitionTarget;
import graph.compile.planning.partition.cost.AcceleratorPartitionScoreModel;

/**
 * Concrete partition planner invocation resolved from {@code BackendPlanningConfig}.
 */
public record BackendPlanningJob(
        PartitionTarget target,
        PartitionPlannerStrategy strategy,
        AcceleratorPartitionScoreModel.PlannerPolicy policy,
        PartitionSourcePolicy sourcePolicy,
        CpuRegionConfig cpuRegionConfig,
        String reason
) {
    public BackendPlanningJob {
        target = target == null ? PartitionTarget.NONE : target;
        strategy = strategy == null ? PartitionPlannerStrategy.GREEDY_MAX_REGION : strategy;
        policy = policy == null
                ? new AcceleratorPartitionScoreModel.PlannerPolicy(1, 1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
                : policy;
        sourcePolicy = sourcePolicy == null ? PartitionSourcePolicy.TARGET_BACKEND_ONLY : sourcePolicy;
        cpuRegionConfig = cpuRegionConfig == null ? CpuRegionConfig.defaults() : cpuRegionConfig;
        reason = reason == null ? "" : reason;
    }
}
