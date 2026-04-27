package graph.optimizer.partition;

public record PartitionValue(
        PartitionValueRef ref,
        int producerNodeId
) {
    public PartitionValue {
        if (ref == null) {
            throw new IllegalArgumentException("ref cannot be null");
        }
        if (producerNodeId < 0) {
            throw new IllegalArgumentException("producerNodeId must be >= 0");
        }
    }
}
