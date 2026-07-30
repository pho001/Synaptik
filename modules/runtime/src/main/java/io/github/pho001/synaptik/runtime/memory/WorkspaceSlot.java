package io.github.pho001.synaptik.runtime.memory;

/**
 * Identifies one workspace slot within a prepared-memory-plan identity domain.
 *
 * <p>This deeply immutable value lets reusable prepared state name a workspace position without
 * retaining a backend-analysis requirement or per-run storage. The valid component range is zero
 * through {@link Long#MAX_VALUE}, with no reserved sentinel. The owning plan is not stored in the
 * record, so callers must interpret the number only in that plan's context. Another plan may reuse
 * the same number without identifying the same slot.
 *
 * <p>A workspace slot is nominally distinct from {@link BufferSlot} and from a Prepare
 * analysis-local workspace requirement identity, even when their numeric values match. It is not
 * physical storage, an address, an allocation, a device or residency fact, or a resource handle.
 * Construction allocates, acquires, retains, and releases none of those resources. Ordinary
 * nominal record equality and hashing compare the exact numeric component only with another
 * {@code WorkspaceSlot}; record text is diagnostic, not a serialization format.
 *
 * @param value the non-negative plan-local identity; zero is valid and no value is reserved
 */
public record WorkspaceSlot(long value) {
    /**
     * Creates a workspace-slot identity for interpretation in one prepared-memory-plan context.
     *
     * @param value the identity value to retain exactly; must be non-negative, zero is valid, and
     *     no value is reserved
     * @throws IllegalArgumentException if {@code value} is negative
     */
    public WorkspaceSlot(long value) {
        if (value < 0) {
            throw new IllegalArgumentException("value must be non-negative");
        }
        this.value = value;
    }

    /**
     * Returns the numeric identity for interpretation in the owning prepared-memory-plan context.
     *
     * <p>The result is not a buffer-slot identity, analysis requirement identity, storage address,
     * physical workspace, device allocation, resource handle, or process-wide identifier.
     *
     * @return the exact stored value in the range zero through {@link Long#MAX_VALUE}
     */
    public long value() {
        return value;
    }
}
