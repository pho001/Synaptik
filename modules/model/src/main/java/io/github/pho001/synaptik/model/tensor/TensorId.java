package io.github.pho001.synaptik.model.tensor;

/**
 * Identifies one public tensor state object independently of graph-local node and value identities.
 *
 * <p>The identifier is an immutable value, not an allocator or registry entry. Its numeric value is
 * meaningful only within the tensor identity policy that assigns it. Later tensor and factory
 * contracts own allocation and uniqueness; this type merely prevents tensor identities from being
 * confused with other identifier domains.</p>
 *
 * @param value non-negative tensor identity value; zero is valid
 */
public record TensorId(long value) {
    /**
     * Creates a tensor identifier with the supplied validated numeric value.
     *
     * @param value non-negative tensor identity value to store exactly; zero is valid and no
     *     sentinel values are reserved
     * @throws IllegalArgumentException if {@code value} is negative
     */
    public TensorId(long value) {
        if (value < 0) {
            throw new IllegalArgumentException("value must be non-negative");
        }
        this.value = value;
    }

    /**
     * Returns the numeric tensor identity value.
     *
     * <p>The result is not a graph node ID, graph value ID, storage address, or globally unique
     * process identifier.</p>
     *
     * @return the stored non-negative identity value
     */
    public long value() {
        return value;
    }
}
