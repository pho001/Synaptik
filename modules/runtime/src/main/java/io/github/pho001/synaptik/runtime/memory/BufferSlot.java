package io.github.pho001.synaptik.runtime.memory;

/**
 * Identifies one buffer slot within a prepared-memory-plan identity domain.
 *
 * <p>This deeply immutable value lets reusable prepared state name a buffer position without
 * retaining per-run storage. The valid component range is zero through {@link Long#MAX_VALUE},
 * with no reserved sentinel. The owning plan is not stored in the record, so callers must
 * interpret the number only in that plan's context. Another plan may reuse the same number without
 * identifying the same slot.
 *
 * <p>A buffer slot is deliberately distinct from a compile-time graph {@code ValueId}. It is not
 * physical storage, an address, an allocation, a device or residency fact, or a resource handle.
 * Construction allocates, acquires, retains, and releases none of those resources. Ordinary
 * record equality and hashing compare the exact numeric component; record text is diagnostic, not
 * a serialization format.
 *
 * @param value the non-negative plan-local identity; zero is valid and no value is reserved
 */
public record BufferSlot(long value) {
    /**
     * Creates a buffer-slot identity for interpretation in one prepared-memory-plan context.
     *
     * @param value the identity value to retain exactly; must be non-negative, zero is valid, and
     *     no value is reserved
     * @throws IllegalArgumentException if {@code value} is negative
     */
    public BufferSlot(long value) {
        if (value < 0) {
            throw new IllegalArgumentException("value must be non-negative");
        }
        this.value = value;
    }

    /**
     * Returns the numeric identity for interpretation in the owning prepared-memory-plan context.
     *
     * <p>The result is not a graph value identity, storage address, physical buffer, device
     * allocation, resource handle, or process-wide identifier.
     *
     * @return the exact stored value in the range zero through {@link Long#MAX_VALUE}
     */
    public long value() {
        return value;
    }
}
