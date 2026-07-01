package planning.partition;

import planning.value.GraphValueRef;

import config.optimizer.CpuPartitionConfig;
import graph.model.CompiledNode;
import planning.partition.cost.AcceleratorPartitionScoreModel;

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
 * @param capability backend partition capability
 * @param sourcePolicy backend-intent eligibility for candidate source nodes
 * @param requiredMaterializedValueRefs values that must remain materialized across partition boundaries
 * @param cpuPartitionConfig CPU partition policy when {@link #strategy()} is
 *                        {@link PartitionPlannerStrategy#CPU_NATURAL_EXECUTION_PARTITION}
 * @param costPreset static materialization cost preset for scored planning
 */
public record PartitionPlanningRequest(
        PartitionPlannerStrategy strategy,
        PartitionTarget target,
        PartitionPlanningContext context,
        AcceleratorPartitionScoreModel.PlannerPolicy policy,
        BackendPartitionCapability capability,
        PartitionSourcePolicy sourcePolicy,
        Set<GraphValueRef> requiredMaterializedValueRefs,
        CpuPartitionConfig cpuPartitionConfig,
        AcceleratorPartitionScoreModel.StaticCostPreset costPreset
) {
    public PartitionPlanningRequest(
            PartitionPlannerStrategy strategy,
            PartitionTarget target,
            PartitionPlanningContext context,
            AcceleratorPartitionScoreModel.PlannerPolicy policy,
            BackendPartitionCapability capability,
            Set<GraphValueRef> requiredMaterializedValueRefs
    ) {
        this(
                strategy,
                target,
                context,
                policy,
                capability,
                PartitionSourcePolicy.TARGET_BACKEND_ONLY,
                requiredMaterializedValueRefs,
                CpuPartitionConfig.defaults(),
                AcceleratorPartitionScoreModel.StaticCostPreset.conservative()
        );
    }

    public PartitionPlanningRequest(
            PartitionPlannerStrategy strategy,
            PartitionTarget target,
            PartitionPlanningContext context,
            AcceleratorPartitionScoreModel.PlannerPolicy policy,
            BackendPartitionCapability capability,
            PartitionSourcePolicy sourcePolicy,
            Set<GraphValueRef> requiredMaterializedValueRefs,
            CpuPartitionConfig cpuPartitionConfig
    ) {
        this(
                strategy,
                target,
                context,
                policy,
                capability,
                sourcePolicy,
                requiredMaterializedValueRefs,
                cpuPartitionConfig,
                AcceleratorPartitionScoreModel.StaticCostPreset.conservative()
        );
    }

    public PartitionPlanningRequest(
            PartitionPlannerStrategy strategy,
            PartitionTarget target,
            PartitionPlanningContext context,
            AcceleratorPartitionScoreModel.PlannerPolicy policy,
            BackendPartitionCapability capability,
            Set<GraphValueRef> requiredMaterializedValueRefs,
            CpuPartitionConfig cpuPartitionConfig
    ) {
        this(
                strategy,
                target,
                context,
                policy,
                capability,
                PartitionSourcePolicy.TARGET_BACKEND_ONLY,
                requiredMaterializedValueRefs,
                cpuPartitionConfig,
                AcceleratorPartitionScoreModel.StaticCostPreset.conservative()
        );
    }

    public PartitionPlanningRequest {
        strategy = strategy == null ? PartitionPlannerStrategy.GREEDY_MAX_PARTITION : strategy;
        target = target == null ? PartitionTarget.NONE : target;
        context = Objects.requireNonNull(context, "context cannot be null");
        policy = policy == null ? AcceleratorPartitionScoreModel.PlannerPolicy.defaults() : policy;
        capability = Objects.requireNonNull(capability, "capability cannot be null");
        sourcePolicy = sourcePolicy == null ? PartitionSourcePolicy.TARGET_BACKEND_ONLY : sourcePolicy;
        requiredMaterializedValueRefs = Set.copyOf(requiredMaterializedValueRefs == null ? Set.of() : new LinkedHashSet<>(requiredMaterializedValueRefs));
        cpuPartitionConfig = cpuPartitionConfig == null ? CpuPartitionConfig.defaults() : cpuPartitionConfig;
        costPreset = costPreset == null ? AcceleratorPartitionScoreModel.StaticCostPreset.conservative() : costPreset;
    }

    /**
     * Returns whether a compiled node may be considered as part of this request's target candidate.
     *
     * @param node compiled node to check
     * @return true when the node's current backend intent is compatible with this planning request
     */
    public boolean canConsiderNode(CompiledNode node) {
        return sourcePolicy.accepts(target, node);
    }
}
