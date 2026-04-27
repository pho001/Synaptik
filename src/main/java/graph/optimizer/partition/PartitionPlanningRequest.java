package graph.optimizer.partition;

import graph.optimizer.partition.cost.AcceleratorPartitionScoreModel;

import java.util.LinkedHashSet;
import java.util.Objects;
import java.util.Set;

public record PartitionPlanningRequest(
        PartitionPlannerStrategy strategy,
        PartitionTarget target,
        PartitionPlanningContext context,
        AcceleratorPartitionScoreModel.PlannerPolicy policy,
        RegionLegalityAdapter adapter,
        Set<PartitionValueRef> requiredMaterializedValueRefs
) {
    public PartitionPlanningRequest {
        strategy = strategy == null ? PartitionPlannerStrategy.GREEDY_MAX_REGION : strategy;
        target = target == null ? PartitionTarget.NONE : target;
        context = Objects.requireNonNull(context, "context cannot be null");
        policy = policy == null ? AcceleratorPartitionScoreModel.PlannerPolicy.defaults() : policy;
        adapter = Objects.requireNonNull(adapter, "adapter cannot be null");
        requiredMaterializedValueRefs = Set.copyOf(requiredMaterializedValueRefs == null ? Set.of() : new LinkedHashSet<>(requiredMaterializedValueRefs));
    }
}
