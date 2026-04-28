package backend.accelerator.dag;

import java.util.List;

/**
 * Lowered accelerator DAG passed to native graph bridges.
 *
 * @param externalInputs runtime graph values consumed by the DAG
 * @param nodes topologically ordered lowered accelerator nodes
 * @param outputNodeIndices indices into {@code nodes} that produce partition outputs
 * @param outputNodeIds compiled-node ids corresponding to partition outputs
 */
public record AcceleratorDagSpec(
        List<AcceleratorDagInput> externalInputs,
        List<AcceleratorDagNode> nodes,
        List<Integer> outputNodeIndices,
        List<Integer> outputNodeIds
) {
    public AcceleratorDagSpec {
        externalInputs = List.copyOf(externalInputs == null ? List.of() : externalInputs);
        nodes = List.copyOf(nodes == null ? List.of() : nodes);
        outputNodeIndices = List.copyOf(outputNodeIndices == null ? List.of() : outputNodeIndices);
        outputNodeIds = List.copyOf(outputNodeIds == null ? List.of() : outputNodeIds);
        if (nodes.isEmpty()) {
            throw new IllegalArgumentException("nodes cannot be empty");
        }
        if (outputNodeIndices.isEmpty()) {
            throw new IllegalArgumentException("outputNodeIndices cannot be empty");
        }
        if (outputNodeIds.size() != outputNodeIndices.size()) {
            throw new IllegalArgumentException("outputNodeIds and outputNodeIndices must have same size");
        }
        for (int outputNodeIndex : outputNodeIndices) {
            if (outputNodeIndex < 0 || outputNodeIndex >= nodes.size()) {
                throw new IllegalArgumentException("outputNodeIndex must point inside nodes");
            }
        }
    }
}
