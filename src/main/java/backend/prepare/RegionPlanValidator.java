package backend.prepare;

import backend.lowering.region.RegionExecutionPlan;
import graph.model.CompiledNode;

import java.util.HashSet;
import java.util.Set;

/**
 * Shared validation for executable region boundary invariants.
 */
public final class RegionPlanValidator {
    private RegionPlanValidator() {
    }

    public static void requireBoundaryCoverage(RegionExecutionPlan plan, BackendPrepareContext context) {
        if (plan == null || context == null) {
            return;
        }
        Set<Integer> regionNodes = new HashSet<>(plan.orderedNodeIds());
        Set<Integer> boundaryNodes = new HashSet<>(plan.boundaryOutputNodeIds());
        for (int nodeId : plan.orderedNodeIds()) {
            boolean hasSelectedConsumer = false;
            boolean hasExternalConsumer = false;
            for (CompiledNode consumer : context.consumersFor(nodeId)) {
                if (consumer == null) {
                    continue;
                }
                if (regionNodes.contains(consumer.id())) {
                    hasSelectedConsumer = true;
                } else {
                    hasExternalConsumer = true;
                }
            }
            if (hasExternalConsumer && !boundaryNodes.contains(nodeId)) {
                throw new IllegalStateException("Region " + plan.regionId()
                        + " has external consumer for non-boundary nodeId=" + nodeId);
            }
            if (!hasSelectedConsumer && !boundaryNodes.contains(nodeId)) {
                throw new IllegalStateException("Region " + plan.regionId()
                        + " has terminal non-boundary nodeId=" + nodeId);
            }
        }
    }
}
