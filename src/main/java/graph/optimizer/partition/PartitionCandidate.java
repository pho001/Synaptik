package graph.optimizer.partition;

import java.util.List;

public record PartitionCandidate(
        int computeNodeId,
        List<Integer> orderedNodeIds,
        List<Integer> externalInputIds,
        List<Integer> outputNodeIds,
        int anchorNodeId
) {
    public PartitionCandidate {
        orderedNodeIds = List.copyOf(orderedNodeIds == null ? List.of() : orderedNodeIds);
        externalInputIds = List.copyOf(externalInputIds == null ? List.of() : externalInputIds);
        outputNodeIds = List.copyOf(outputNodeIds == null ? List.of() : outputNodeIds);
    }
}
