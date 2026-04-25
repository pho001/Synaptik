package graph.optimizer.partition.model;

import java.util.List;

public record AcceleratorDagSpec(
        List<AcceleratorDagInput> externalInputs,
        List<AcceleratorDagNode> nodes,
        int outputNodeIndex,
        int outputNodeId
) {
    public AcceleratorDagSpec {
        externalInputs = List.copyOf(externalInputs == null ? List.of() : externalInputs);
        nodes = List.copyOf(nodes == null ? List.of() : nodes);
        if (nodes.isEmpty()) {
            throw new IllegalArgumentException("nodes cannot be empty");
        }
        if (outputNodeIndex < 0 || outputNodeIndex >= nodes.size()) {
            throw new IllegalArgumentException("outputNodeIndex must point inside nodes");
        }
    }
}
