package graph.optimizer.partition;

import java.util.List;

public record AcceleratorStructuralCandidate(
        int computeNodeId,
        List<Integer> orderedNodeIds,
        List<Integer> externalInputIds,
        int anchorNodeId
) {
    public AcceleratorStructuralCandidate {
        orderedNodeIds = List.copyOf(orderedNodeIds == null ? List.of() : orderedNodeIds);
        externalInputIds = List.copyOf(externalInputIds == null ? List.of() : externalInputIds);
    }
}
