package graph.optimizer.partition;

import backend.ComputeBackend;

import java.util.List;
import java.util.Set;

/**
 * Partition candidate together with backend compatibility and an optional lowered plan.
 *
 * @param partition accepted partition
 * @param compatibleBackends backends that can execute the partition
 * @param plan backend plan attached to the partition, if lowering succeeded
 */
public record BackendCandidatePartition(
        Partition partition,
        Set<ComputeBackend> compatibleBackends,
        PartitionPlan plan
) {
    public BackendCandidatePartition {
        if (partition == null) {
            throw new IllegalArgumentException("partition cannot be null");
        }
        compatibleBackends = Set.copyOf(compatibleBackends == null ? Set.of() : compatibleBackends);
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
