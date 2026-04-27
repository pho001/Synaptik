package graph.optimizer.partition;

import graph.CompiledNode;

import java.util.Set;

public final class UnsupportedRegionLegalityAdapter implements RegionLegalityAdapter {
    private final PartitionTarget target;

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
