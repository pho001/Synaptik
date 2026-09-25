package io.github.pho001.synaptik.trace.id;

/**
 * Identifies one prepared-unit invocation for diagnostic correlation within a producer-defined
 * trace stream.
 *
 * <p>The producer assigns this immutable trace-local value and owns its uniqueness, lifetime, and
 * mapping from any producer-domain invocation. The numeric value need not match a producer
 * identifier and has no process-wide or cross-stream uniqueness guarantee. This identifier does
 * not contain run state, input or output values, thread identity, or timing, and it performs no
 * allocation, mapping, or serialization.</p>
 *
 * @param value non-negative producer-supplied trace-local invocation correlation value; zero is
 *     valid
 */
public record TraceInvocationId(long value) {
    /**
     * Creates a trace-local invocation correlation identifier.
     *
     * @param value non-negative correlation value to retain exactly; zero is valid and no sentinel
     *     is reserved
     * @throws IllegalArgumentException if {@code value} is negative
     */
    public TraceInvocationId(long value) {
        if (value < 0) {
            throw new IllegalArgumentException("value must be non-negative");
        }
        this.value = value;
    }

    /**
     * Returns the producer-supplied trace-local invocation correlation value.
     *
     * @return the exact stored non-negative value
     */
    public long value() {
        return value;
    }
}
