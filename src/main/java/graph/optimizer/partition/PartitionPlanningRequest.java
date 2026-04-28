package graph.optimizer.partition;

import config.optimizer.CpuRegionConfig;
import graph.optimizer.partition.cost.AcceleratorPartitionScoreModel;

import java.util.LinkedHashSet;
import java.util.Objects;
import java.util.Set;

/**
 * Input bundle for a partition planner.
 *
 * @param strategy planner strategy requested by configuration
 * @param target backend target to plan for
 * @param context compiled graph context
 * @param policy scoring and search policy
 * @param adapter backend legality and lowering adapter
 * @param requiredMaterializedValueRefs values that must remain materialized across region boundaries
 * @param cpuRegionConfig CPU region policy when {@link #strategy()} is
 *                        {@link PartitionPlannerStrategy#CPU_NATURAL_EXECUTION_REGION}
 */
public record PartitionPlanningRequest(
        PartitionPlannerStrategy strategy,
        PartitionTarget target,
        PartitionPlanningContext context,
        AcceleratorPartitionScoreModel.PlannerPolicy policy,
        RegionLegalityAdapter adapter,
        Set<PartitionValueRef> requiredMaterializedValueRefs,
        CpuRegionConfig cpuRegionConfig
) {
    public PartitionPlanningRequest(
            PartitionPlannerStrategy strategy,
            PartitionTarget target,
            PartitionPlanningContext context,
            AcceleratorPartitionScoreModel.PlannerPolicy policy,
            RegionLegalityAdapter adapter,
            Set<PartitionValueRef> requiredMaterializedValueRefs
    ) {
        this(strategy, target, context, policy, adapter, requiredMaterializedValueRefs, CpuRegionConfig.defaults());
    }

    public PartitionPlanningRequest {
        strategy = strategy == null ? PartitionPlannerStrategy.GREEDY_MAX_REGION : strategy;
        target = target == null ? PartitionTarget.NONE : target;
        context = Objects.requireNonNull(context, "context cannot be null");
        policy = policy == null ? AcceleratorPartitionScoreModel.PlannerPolicy.defaults() : policy;
        adapter = Objects.requireNonNull(adapter, "adapter cannot be null");
        requiredMaterializedValueRefs = Set.copyOf(requiredMaterializedValueRefs == null ? Set.of() : new LinkedHashSet<>(requiredMaterializedValueRefs));
        cpuRegionConfig = cpuRegionConfig == null ? CpuRegionConfig.defaults() : cpuRegionConfig;
    }
}
