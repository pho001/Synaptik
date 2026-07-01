package prepare.validation;

import backend.lowering.partition.BackendPartitionExecutionPlan;
import graph.model.CompiledNode;
import prepare.context.BackendPrepareContext;

import java.util.HashSet;
import java.util.Set;

/**
 * Shared validation for executable partition boundary invariants.
 */
public final class BackendPartitionExecutionPlanValidator {
    private BackendPartitionExecutionPlanValidator() {
    }

    public static void requireBoundaryCoverage(BackendPartitionExecutionPlan plan, BackendPrepareContext context) {
        if (plan == null || context == null) {
            return;
        }
        Set<Integer> partitionNodes = new HashSet<>(plan.orderedNodeIds());
        Set<Integer> boundaryNodes = new HashSet<>(plan.boundaryOutputNodeIds());
        for (int nodeId : plan.orderedNodeIds()) {
            boolean hasSelectedConsumer = false;
            boolean hasExternalConsumer = false;
            for (CompiledNode consumer : context.consumersFor(nodeId)) {
                if (consumer == null) {
                    continue;
                }
                if (partitionNodes.contains(consumer.id())) {
                    hasSelectedConsumer = true;
                } else {
                    hasExternalConsumer = true;
                }
            }
            if (hasExternalConsumer && !boundaryNodes.contains(nodeId)) {
                throw new IllegalStateException("Partition execution plan " + plan.executionPlanId()
                        + " has external consumer for non-boundary nodeId=" + nodeId);
            }
            if (!hasSelectedConsumer && !boundaryNodes.contains(nodeId)) {
                throw new IllegalStateException("Partition execution plan " + plan.executionPlanId()
                        + " has terminal non-boundary nodeId=" + nodeId);
            }
        }
    }
}
