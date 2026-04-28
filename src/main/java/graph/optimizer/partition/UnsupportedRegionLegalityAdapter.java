package graph.optimizer.partition;

import graph.CompiledNode;

import java.util.Set;

/**
 * Null-object legality adapter for targets that have no region lowerer.
 *
 * <p>Every support and lowering check rejects the candidate, allowing planners to run without special casing missing
 * backend support.
 */
public final class UnsupportedRegionLegalityAdapter implements RegionLegalityAdapter {
    private final PartitionTarget target;

    /**
     * Creates an adapter for an unsupported target.
     *
     * @param target target to report from {@link #target()}
     */
    public UnsupportedRegionLegalityAdapter(PartitionTarget target) {
        this.target = target == null ? PartitionTarget.NONE : target;
    }

    @Override
    public PartitionTarget target() {
        return target;
    }

    @Override
    public boolean isNodeSupported(CompiledNode node, PartitionPlanningContext context) {
        return false;
    }

    @Override
    public boolean canSeed(CompiledNode node, PartitionPlanningContext context) {
        return false;
    }

    @Override
    public boolean canUseAsExternalInput(
            CompiledNode producer,
            CompiledNode consumer,
            Set<Integer> selectedNodeIds,
            PartitionPlanningContext context
    ) {
        return false;
    }

    @Override
    public PartitionCandidate tryCreateStructuralCandidate(
            Set<Integer> selectedNodeIds,
            PartitionPlanningContext context,
            Set<PartitionValueRef> requiredMaterializedValueRefs
    ) {
        return null;
    }

    @Override
    public PartitionPlan tryCreatePlan(PartitionCandidate candidate, PartitionPlanningContext context) {
        return null;
    }
}
