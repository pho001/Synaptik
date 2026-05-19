package graph.optimizer.partition;

import graph.optimizer.GraphValueRef;

/**
 * Value produced by a node inside a partition.
 *
 * @param ref stable value reference
 * @param producerNodeId node id that produces the value
 */
public record PartitionValue(
        GraphValueRef ref,
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
