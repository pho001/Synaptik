package planning.backend;

import config.optimizer.CpuPartitionConfig;
import planning.partition.PartitionPlannerStrategy;
import planning.partition.PartitionSourcePolicy;
import planning.partition.PartitionTarget;
import planning.partition.cost.AcceleratorPartitionScoreModel;

/**
 * Concrete partition planner invocation resolved from {@code BackendPlanningConfig}.
 */
public record BackendPlanningJob(
        PartitionTarget target,
        PartitionPlannerStrategy strategy,
        AcceleratorPartitionScoreModel.PlannerPolicy policy,
        PartitionSourcePolicy sourcePolicy,
        CpuPartitionConfig cpuPartitionConfig,
        String reason
) {
    public BackendPlanningJob {
        target = target == null ? PartitionTarget.NONE : target;
        strategy = strategy == null ? PartitionPlannerStrategy.GREEDY_MAX_PARTITION : strategy;
        policy = policy == null
                ? new AcceleratorPartitionScoreModel.PlannerPolicy(1, 1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
                : policy;
        sourcePolicy = sourcePolicy == null ? PartitionSourcePolicy.TARGET_BACKEND_ONLY : sourcePolicy;
        cpuPartitionConfig = cpuPartitionConfig == null ? CpuPartitionConfig.defaults() : cpuPartitionConfig;
        reason = reason == null ? "" : reason;
    }
}
