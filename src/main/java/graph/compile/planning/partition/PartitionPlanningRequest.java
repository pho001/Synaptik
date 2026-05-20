package graph.compile.planning.partition;

import graph.compile.planning.value.GraphValueRef;

import config.optimizer.CpuRegionConfig;
import config.optimizer.MetalTransferModel;
import graph.CompiledNode;
import graph.compile.planning.partition.cost.AcceleratorPartitionScoreModel;

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
 * @param sourcePolicy backend-intent eligibility for candidate source nodes
 * @param requiredMaterializedValueRefs values that must remain materialized across region boundaries
 * @param cpuRegionConfig CPU region policy when {@link #strategy()} is
 *                        {@link PartitionPlannerStrategy#CPU_NATURAL_EXECUTION_REGION}
 * @param metalTransferModel transfer-cost model for scored Metal planning
 */
public record PartitionPlanningRequest(
        PartitionPlannerStrategy strategy,
        PartitionTarget target,
        PartitionPlanningContext context,
        AcceleratorPartitionScoreModel.PlannerPolicy policy,
        RegionLegalityAdapter adapter,
        PartitionSourcePolicy sourcePolicy,
        Set<GraphValueRef> requiredMaterializedValueRefs,
        CpuRegionConfig cpuRegionConfig,
        MetalTransferModel metalTransferModel
) {
    public PartitionPlanningRequest(
            PartitionPlannerStrategy strategy,
            PartitionTarget target,
            PartitionPlanningContext context,
            AcceleratorPartitionScoreModel.PlannerPolicy policy,
            RegionLegalityAdapter adapter,
            Set<GraphValueRef> requiredMaterializedValueRefs
    ) {
        this(
                strategy,
                target,
                context,
                policy,
                adapter,
                PartitionSourcePolicy.TARGET_BACKEND_ONLY,
                requiredMaterializedValueRefs,
                CpuRegionConfig.defaults(),
                MetalTransferModel.CONSERVATIVE
        );
    }

    public PartitionPlanningRequest(
            PartitionPlannerStrategy strategy,
            PartitionTarget target,
            PartitionPlanningContext context,
            AcceleratorPartitionScoreModel.PlannerPolicy policy,
            RegionLegalityAdapter adapter,
            PartitionSourcePolicy sourcePolicy,
            Set<GraphValueRef> requiredMaterializedValueRefs,
            CpuRegionConfig cpuRegionConfig
    ) {
        this(
                strategy,
                target,
                context,
                policy,
                adapter,
                sourcePolicy,
                requiredMaterializedValueRefs,
                cpuRegionConfig,
                MetalTransferModel.CONSERVATIVE
        );
    }

    public PartitionPlanningRequest(
            PartitionPlannerStrategy strategy,
            PartitionTarget target,
            PartitionPlanningContext context,
            AcceleratorPartitionScoreModel.PlannerPolicy policy,
            RegionLegalityAdapter adapter,
            Set<GraphValueRef> requiredMaterializedValueRefs,
            CpuRegionConfig cpuRegionConfig
    ) {
        this(
                strategy,
                target,
                context,
                policy,
                adapter,
                PartitionSourcePolicy.TARGET_BACKEND_ONLY,
                requiredMaterializedValueRefs,
                cpuRegionConfig,
                MetalTransferModel.CONSERVATIVE
        );
    }

    public PartitionPlanningRequest {
        strategy = strategy == null ? PartitionPlannerStrategy.GREEDY_MAX_REGION : strategy;
        target = target == null ? PartitionTarget.NONE : target;
        context = Objects.requireNonNull(context, "context cannot be null");
        policy = policy == null ? AcceleratorPartitionScoreModel.PlannerPolicy.defaults() : policy;
        adapter = Objects.requireNonNull(adapter, "adapter cannot be null");
        sourcePolicy = sourcePolicy == null ? PartitionSourcePolicy.TARGET_BACKEND_ONLY : sourcePolicy;
        requiredMaterializedValueRefs = Set.copyOf(requiredMaterializedValueRefs == null ? Set.of() : new LinkedHashSet<>(requiredMaterializedValueRefs));
        cpuRegionConfig = cpuRegionConfig == null ? CpuRegionConfig.defaults() : cpuRegionConfig;
        metalTransferModel = metalTransferModel == null ? MetalTransferModel.CONSERVATIVE : metalTransferModel;
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
