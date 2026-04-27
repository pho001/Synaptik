package graph.optimizer.partition;

import graph.CompiledNode;

import java.util.Set;

public interface RegionLegalityAdapter {
    PartitionTarget target();

    boolean isNodeSupported(CompiledNode node, PartitionPlanningContext context);

    boolean canSeed(CompiledNode node, PartitionPlanningContext context);

    boolean canUseAsExternalInput(
            CompiledNode producer,
            CompiledNode consumer,
            Set<Integer> selectedNodeIds,
            PartitionPlanningContext context
    );

    PartitionCandidate tryCreateStructuralCandidate(
            Set<Integer> selectedNodeIds,
            PartitionPlanningContext context,
            Set<PartitionValueRef> requiredMaterializedValueRefs
    );

    PartitionPlan tryCreatePlan(
            PartitionCandidate candidate,
            PartitionPlanningContext context
    );
}
