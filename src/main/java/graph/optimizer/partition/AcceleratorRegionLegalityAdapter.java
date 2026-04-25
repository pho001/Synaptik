package graph.optimizer.partition;

import backend.prepare.BackendPrepareContext;
import graph.CompiledNode;

import java.util.Set;

public interface AcceleratorRegionLegalityAdapter {
    AcceleratorTarget target();

    boolean isNodeSupported(CompiledNode node, BackendPrepareContext context);

    boolean canSeed(CompiledNode node, BackendPrepareContext context);

    boolean canUseAsExternalInput(
            CompiledNode producer,
            CompiledNode consumer,
            Set<Integer> selectedNodeIds,
            BackendPrepareContext context
    );

    AcceleratorStructuralCandidate tryCreateStructuralCandidate(
            Set<Integer> selectedNodeIds,
            BackendPrepareContext context
    );

    AcceleratorPartitionPlan tryCreatePlan(
            AcceleratorStructuralCandidate candidate,
            BackendPrepareContext context
    );
}
