package graph.compile.planning.partition;

import backend.ComputeBackend;
import graph.CompiledNode;

/**
 * Controls which compiled-node backend intents a partition planner may consider for a target.
 *
 * <p>The target backend still describes the backend that would own the accepted partition. The source policy describes
 * whether candidates must already carry that backend intent, or whether CPU nodes may be considered as accelerator
 * offload candidates without mutating the compiled graph's original backend assignment.</p>
 */
public enum PartitionSourcePolicy {
    /**
     * Only nodes whose compiled backend already matches the requested partition target may be considered.
     */
    TARGET_BACKEND_ONLY,

    /**
     * CPU nodes and nodes already assigned to the requested target may be considered.
     *
     * <p>This is used by graph-level offload planning: the graph remains CPU-owned by default, while accelerator
     * planning evaluates legal accelerator ownership regions as candidates.</p>
     */
    CPU_OR_TARGET_BACKEND;

    /**
     * Returns whether {@code node} is eligible for planning against {@code target}.
     *
     * @param target partition target being planned
     * @param node compiled node being considered
     * @return true when the node may be included in a candidate for the target
     */
    public boolean accepts(PartitionTarget target, CompiledNode node) {
        if (target == null || target.isNone() || node == null) {
            return false;
        }
        ComputeBackend targetBackend = target.backend();
        if (targetBackend == null) {
            return false;
        }
        return switch (this) {
            case TARGET_BACKEND_ONLY -> node.backend() == targetBackend;
            case CPU_OR_TARGET_BACKEND -> node.backend() == targetBackend || node.backend() == ComputeBackend.CPU;
        };
    }
}
