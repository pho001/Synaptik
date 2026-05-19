package graph.optimizer.partition;

import backend.ComputeBackend;
import graph.compile.BackendPlanningDiagnostic;

import java.util.List;
import java.util.Set;

/**
 * Accepted partition together with its executable backend plan.
 *
 * <p>This is the compile-to-prepare contract for backend-owned regions. It keeps the structural partition and the
 * attached backend plan together so later stages do not have to reconcile parallel lists or infer ownership through
 * object identity.
 *
 * @param partition accepted partition
 * @param plan backend plan attached to the partition
 * @param compatibleBackends backends that can execute the partition
 * @param diagnostics diagnostics associated with this planned partition
 */
public record PlannedPartition(
        Partition partition,
        PartitionPlan plan,
        Set<ComputeBackend> compatibleBackends,
        List<BackendPlanningDiagnostic> diagnostics
) {
    public PlannedPartition {
        if (partition == null) {
            throw new IllegalArgumentException("partition cannot be null");
        }
        compatibleBackends = Set.copyOf(compatibleBackends == null ? Set.of() : compatibleBackends);
        diagnostics = List.copyOf(diagnostics == null ? List.of() : diagnostics);
    }

    public PlannedPartition(
            Partition partition,
            PartitionPlan plan,
            Set<ComputeBackend> compatibleBackends
    ) {
        this(partition, plan, compatibleBackends, List.of());
    }

    /**
     * Returns the anchor node id from the underlying partition.
     *
     * @return anchor node id
     */
    public int anchorNodeId() {
        return partition.anchorSeedNodeId();
    }

    /**
     * Returns node ids from the underlying partition.
     *
     * @return ordered partition node ids
     */
    public List<Integer> nodeIds() {
        return partition.orderedNodeIds();
    }
}
