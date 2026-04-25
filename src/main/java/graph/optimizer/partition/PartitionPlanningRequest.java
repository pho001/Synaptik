package graph.optimizer.partition;

import backend.prepare.BackendPrepareContext;
import graph.optimizer.partition.cost.AcceleratorPartitionScoreModel;

import java.util.Objects;

public record PartitionPlanningRequest(
        PartitionPlannerStrategy strategy,
        AcceleratorTarget target,
        BackendPrepareContext context,
        AcceleratorPartitionScoreModel.PlannerPolicy policy,
        AcceleratorRegionLegalityAdapter adapter
) {
    public PartitionPlanningRequest {
        strategy = strategy == null ? PartitionPlannerStrategy.GREEDY_MAX_REGION : strategy;
        target = target == null ? AcceleratorTarget.NONE : target;
        context = Objects.requireNonNull(context, "context cannot be null");
        policy = policy == null ? AcceleratorPartitionScoreModel.PlannerPolicy.defaults() : policy;
        adapter = Objects.requireNonNull(adapter, "adapter cannot be null");
    }
}
