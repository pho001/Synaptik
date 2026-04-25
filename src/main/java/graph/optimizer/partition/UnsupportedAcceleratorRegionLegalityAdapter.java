package graph.optimizer.partition;

import backend.prepare.BackendPrepareContext;
import graph.CompiledNode;

import java.util.Set;

public final class UnsupportedAcceleratorRegionLegalityAdapter implements AcceleratorRegionLegalityAdapter {
    private final AcceleratorTarget target;

    public UnsupportedAcceleratorRegionLegalityAdapter(AcceleratorTarget target) {
        this.target = target == null ? AcceleratorTarget.NONE : target;
    }

    @Override
    public AcceleratorTarget target() {
        return target;
    }

    @Override
    public boolean isNodeSupported(CompiledNode node, BackendPrepareContext context) {
        return false;
    }

    @Override
    public boolean canSeed(CompiledNode node, BackendPrepareContext context) {
        return false;
    }

    @Override
    public boolean canUseAsExternalInput(
            CompiledNode producer,
            CompiledNode consumer,
            Set<Integer> selectedNodeIds,
            BackendPrepareContext context
    ) {
        return false;
    }

    @Override
    public AcceleratorStructuralCandidate tryCreateStructuralCandidate(Set<Integer> selectedNodeIds, BackendPrepareContext context) {
        return null;
    }

    @Override
    public AcceleratorPartitionPlan tryCreatePlan(AcceleratorStructuralCandidate candidate, BackendPrepareContext context) {
        return null;
    }
}
