package graph.optimizer.partition;

import backend.ComputeBackend;

import java.util.List;
import java.util.Set;

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

    public int anchorNodeId() {
        return partition.anchorSeedNodeId();
    }

    public List<Integer> nodeIds() {
        return partition.orderedNodeIds();
    }
}
