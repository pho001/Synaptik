package io.github.pho001.synaptik.trace;

/**
 * Identifies one diagnostic event within a producer-defined trace stream.
 *
 * <p>This immutable value does not allocate identifiers or establish process-wide uniqueness. The
 * producer assigns the value and defines the stream within which it is unique. Equality, hashing,
 * and text use ordinary record component semantics.</p>
 *
 * @param value non-negative producer-supplied event identity; zero is valid
 */
public record TraceEventId(long value) {
    /**
     * Creates an event identifier from a producer-supplied value.
     *
     * @param value non-negative event identity to retain exactly; zero is valid and no sentinel is
     *     reserved
     * @throws IllegalArgumentException if {@code value} is negative
     */
    public TraceEventId(long value) {
        if (value < 0) {
            throw new IllegalArgumentException("value must be non-negative");
        }
        this.value = value;
    }

    /**
     * Returns the producer-supplied event identity.
     *
     * @return the exact stored non-negative producer-supplied value
     */
    public long value() {
        return value;
    }
}
