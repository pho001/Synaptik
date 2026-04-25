package graph.optimizer.partition;

import backend.ComputeBackend;

import java.util.List;
import java.util.Set;

public record AcceleratorCandidatePartition(
        int anchorNodeId,
        List<Integer> nodeIds,
        Set<ComputeBackend> compatibleBackends,
        AcceleratorPartitionPlan plan
) {
    public AcceleratorCandidatePartition {
        nodeIds = List.copyOf(nodeIds == null ? List.of() : nodeIds);
        compatibleBackends = Set.copyOf(compatibleBackends == null ? Set.of() : compatibleBackends);
    }
}
