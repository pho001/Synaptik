package graph.optimizer.partition;

/**
 * Producer-consumer edge represented by compiled node ids.
 *
 * @param producerNodeId producer node id
 * @param consumerNodeId consumer node id
 */
public record PartitionEdge(
        int producerNodeId,
        int consumerNodeId
) {
    public PartitionEdge {
        if (producerNodeId < 0 || consumerNodeId < 0) {
            throw new IllegalArgumentException("edge node ids must be >= 0");
        }
    }
}
