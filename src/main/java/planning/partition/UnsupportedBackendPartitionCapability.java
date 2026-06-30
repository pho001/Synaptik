package planning.partition;

import planning.value.GraphValueRef;

import graph.model.CompiledNode;

import java.util.Set;

/**
 * Null-object partition capability for targets that have no region lowerer.
 *
 * <p>Every support and lowering check rejects the candidate, allowing planners to run without special casing missing
 * backend support.
 */
public final class UnsupportedBackendPartitionCapability implements BackendPartitionCapability {
    private final PartitionTarget target;

    /**
     * Creates a capability for an unsupported target.
     *
     * @param target target to report from {@link #target()}
     */
    public UnsupportedBackendPartitionCapability(PartitionTarget target) {
        this.target = target == null ? PartitionTarget.NONE : target;
    }

    @Override
    public PartitionTarget target() {
        return target;
    }

    @Override
    public boolean canExecute(CompiledNode node, PartitionPlanningContext context) {
        return false;
    }

    @Override
    public boolean canSeed(CompiledNode node, PartitionPlanningContext context) {
        return false;
    }

    @Override
    public boolean canUseExternalInput(
            CompiledNode producer,
            CompiledNode consumer,
            Set<Integer> selectedNodeIds,
            PartitionPlanningContext context
    ) {
        return false;
    }

    @Override
    public PartitionCandidate createCandidate(
            Set<Integer> selectedNodeIds,
            PartitionPlanningContext context,
            Set<GraphValueRef> requiredMaterializedValueRefs
    ) {
        return null;
    }

    @Override
    public PartitionPlan createPlan(PartitionCandidate candidate, PartitionPlanningContext context) {
        return null;
    }
}
