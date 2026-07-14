package io.github.pho001.synaptik.trace.id;

/**
 * Identifies logical graph data for diagnostic correlation within a producer-defined trace stream
 * or correlation domain.
 *
 * <p>The producer assigns this immutable trace-local value and owns its uniqueness, lifetime, and
 * mapping from any producer-domain value identity. The numeric value need not match a producer
 * identifier and has no process-wide or cross-stream uniqueness guarantee. This identifier does
 * not identify a node, public Tensor, storage location, buffer, or runtime slot, and it performs
 * no allocation, mapping, or serialization.</p>
 *
 * <p>Equality, hashing, and diagnostic text use ordinary record component semantics.</p>
 *
 * @param value non-negative producer-supplied trace-local logical-value correlation value; zero
 *     is valid
 */
public record TraceValueId(long value) {
    /**
     * Creates a trace-local logical-value correlation identifier from a producer-supplied value.
     *
     * @param value non-negative correlation value to retain exactly; zero is valid and no sentinel
     *     is reserved
     * @throws IllegalArgumentException if {@code value} is negative
     */
    public TraceValueId(long value) {
        if (value < 0) {
            throw new IllegalArgumentException("value must be non-negative");
        }
        this.value = value;
    }

    /**
     * Returns the producer-supplied trace-local logical-value correlation value.
     *
     * @return the exact stored non-negative value
     */
    public long value() {
        return value;
    }
}
