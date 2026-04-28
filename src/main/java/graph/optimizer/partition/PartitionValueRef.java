package graph.optimizer.partition;

/**
 * Stable reference to a graph value at a partition boundary.
 *
 * @param producerNodeId node id that produces the referenced value
 */
public record PartitionValueRef(
        int producerNodeId
) {
    public PartitionValueRef {
        if (producerNodeId < 0) {
            throw new IllegalArgumentException("producerNodeId must be >= 0");
        }
    }

    /**
     * Creates a value reference for a producer node.
     *
     * @param producerNodeId producer node id
     * @return value reference
     */
    public static PartitionValueRef ofNode(int producerNodeId) {
        return new PartitionValueRef(producerNodeId);
    }
}
