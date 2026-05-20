package graph.compile.planning.partition;

import java.util.List;

/**
 * Backend-neutral candidate selected during partition search.
 *
 * @param computeNodeId representative compute node used by lowerers
 * @param orderedNodeIds selected graph node ids in execution order
 * @param externalInputIds producer node ids outside the candidate
 * @param outputNodeIds node ids whose values leave the candidate
 * @param anchorNodeId node id that seeded the candidate
 */
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
