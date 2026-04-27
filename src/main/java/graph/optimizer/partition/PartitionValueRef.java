package graph.optimizer.partition;

public record PartitionValueRef(
        int producerNodeId
) {
    public PartitionValueRef {
        if (producerNodeId < 0) {
            throw new IllegalArgumentException("producerNodeId must be >= 0");
        }
    }

    public static PartitionValueRef ofNode(int producerNodeId) {
        return new PartitionValueRef(producerNodeId);
    }
}
