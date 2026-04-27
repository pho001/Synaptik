package backend.accelerator.dag;

import java.util.List;

public record AcceleratorSubgraphSpec(
        int computeNodeId,
        List<Integer> orderedNodeIds,
        List<AcceleratorSubgraphOp> ops,
        List<Integer> externalInputNodeIds,
        List<Integer> outputNodeIds
) {
    public AcceleratorSubgraphSpec {
        orderedNodeIds = List.copyOf(orderedNodeIds == null ? List.of() : orderedNodeIds);
        ops = List.copyOf(ops == null ? List.of() : ops);
        externalInputNodeIds = List.copyOf(externalInputNodeIds == null ? List.of() : externalInputNodeIds);
        outputNodeIds = List.copyOf(outputNodeIds == null ? List.of() : outputNodeIds);
        if (orderedNodeIds.isEmpty()) {
            throw new IllegalArgumentException("orderedNodeIds cannot be empty");
        }
        if (!orderedNodeIds.contains(computeNodeId)) {
            throw new IllegalArgumentException("computeNodeId must be part of orderedNodeIds");
        }
        if (ops.size() != orderedNodeIds.size()) {
            throw new IllegalArgumentException("ops and orderedNodeIds must have the same size");
        }
    }
}
