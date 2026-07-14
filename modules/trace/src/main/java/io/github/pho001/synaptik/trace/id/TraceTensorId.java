package io.github.pho001.synaptik.trace.id;

/**
 * Identifies public Tensor state for diagnostic correlation within a producer-defined trace stream
 * or correlation domain.
 *
 * <p>The producer assigns this immutable trace-local value and owns its uniqueness, lifetime, and
 * mapping from any producer-domain Tensor identity. The numeric value need not match a producer
 * identifier and has no process-wide or cross-stream uniqueness guarantee. This identifier does
 * not identify a graph node or logical value, storage address, device allocation, or runtime
 * residency, and it performs no allocation, mapping, or serialization.</p>
 *
 * <p>Equality, hashing, and diagnostic text use ordinary record component semantics.</p>
 *
 * @param value non-negative producer-supplied trace-local public-Tensor correlation value; zero
 *     is valid
 */
public record TraceTensorId(long value) {
    /**
     * Creates a trace-local public-Tensor correlation identifier from a producer-supplied value.
     *
     * @param value non-negative correlation value to retain exactly; zero is valid and no sentinel
     *     is reserved
     * @throws IllegalArgumentException if {@code value} is negative
     */
    public TraceTensorId(long value) {
        if (value < 0) {
            throw new IllegalArgumentException("value must be non-negative");
        }
        this.value = value;
    }

    /**
     * Returns the producer-supplied trace-local public-Tensor correlation value.
     *
     * @return the exact stored non-negative value
     */
    public long value() {
        return value;
    }
}
