package io.github.pho001.synaptik.model.graph;

/**
 * Identifies one logical data value within an owning compiled-graph context.
 *
 * <p>A value identity may address a graph input, intermediate result, or graph output. It is
 * intentionally distinct from {@link NodeId}: a value may have no producing node, one node may
 * produce multiple values, and one value may have multiple consumers. The same numeric value may
 * be reused by another graph; graph construction and compiler contracts own allocation and
 * uniqueness.</p>
 *
 * @param value non-negative graph-local value identity value; zero is valid
 */
public record ValueId(long value) {
    /**
     * Creates a graph-local value identifier with the supplied validated numeric value.
     *
     * @param value non-negative value identity to store exactly; zero is valid and no sentinel
     *     values are reserved
     * @throws IllegalArgumentException if {@code value} is negative
     */
    public ValueId(long value) {
        if (value < 0) {
            throw new IllegalArgumentException("value must be non-negative");
        }
        this.value = value;
    }

    /**
     * Returns the numeric value identity within its owning graph context.
     *
     * <p>The result identifies logical data and is not a memory slot, storage address, tensor ID,
     * or node ID.</p>
     *
     * @return the stored non-negative graph-local identity value
     */
    public long value() {
        return value;
    }
}
