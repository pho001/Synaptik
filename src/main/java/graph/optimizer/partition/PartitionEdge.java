package graph.optimizer.partition;

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
