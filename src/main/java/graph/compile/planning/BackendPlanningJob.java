package graph.compile.planning;

import config.optimizer.CpuRegionConfig;
import config.optimizer.MetalTransferModel;
import graph.optimizer.partition.PartitionPlannerStrategy;
import graph.optimizer.partition.PartitionSourcePolicy;
import graph.optimizer.partition.PartitionTarget;
import graph.optimizer.partition.cost.AcceleratorPartitionScoreModel;

/**
 * Concrete partition planner invocation resolved from {@code BackendPlanningConfig}.
 */
public record BackendPlanningJob(
        PartitionTarget target,
        PartitionPlannerStrategy strategy,
        AcceleratorPartitionScoreModel.PlannerPolicy policy,
        PartitionSourcePolicy sourcePolicy,
        CpuRegionConfig cpuRegionConfig,
        MetalTransferModel metalTransferModel,
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
        metalTransferModel = metalTransferModel == null ? MetalTransferModel.CONSERVATIVE : metalTransferModel;
        reason = reason == null ? "" : reason;
    }
}
