package io.github.pho001.synaptik.model.graph;

/**
 * Identifies one node occurrence within an owning compiled-graph context.
 *
 * <p>A node identity addresses a computation occurrence, not operation semantics or an output
 * value. The same numeric value may be reused by another graph, so callers must interpret this
 * immutable identifier through the graph that owns it. Graph construction and compiler contracts
 * own allocation and uniqueness.</p>
 *
 * @param value non-negative graph-local node identity value; zero is valid
 */
public record NodeId(long value) {
    /**
     * Creates a graph-local node identifier with the supplied validated numeric value.
     *
     * @param value non-negative node identity value to store exactly; zero is valid and no sentinel
     *     values are reserved
     * @throws IllegalArgumentException if {@code value} is negative
     */
    public NodeId(long value) {
        if (value < 0) {
            throw new IllegalArgumentException("value must be non-negative");
        }
        this.value = value;
    }

    /**
     * Returns the numeric node identity value within its owning graph context.
     *
     * @return the stored non-negative graph-local identity value
     */
    public long value() {
        return value;
    }
}
