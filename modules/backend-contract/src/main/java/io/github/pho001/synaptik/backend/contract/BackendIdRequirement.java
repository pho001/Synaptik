package io.github.pho001.synaptik.backend.contract;

/**
 * Requires later eligible ownership to use one exact backend identity.
 *
 * <p>Later ownership targets a {@link BackendId} equal to the stored identity; object identity is
 * not required. The exact non-null reference supplied by the caller is retained. Equality,
 * hashing, and diagnostic text follow ordinary record semantics over that identity.</p>
 *
 * <p>This immutable target does not prove that the backend is registered, available, capable, or
 * preparable and does not evaluate eligibility.</p>
 *
 * @param backendId non-null exact backend identity required for later eligible ownership;
 *     retained by reference
 */
public record BackendIdRequirement(BackendId backendId) implements BackendRequirement {
    /**
     * Creates an exact-backend hard-eligibility target.
     *
     * @param backendId non-null exact backend identity required for later eligible ownership;
     *     retained by reference
     * @throws NullPointerException if {@code backendId} is {@code null}; the exception message is
     *     {@code backendId}
     */
    public BackendIdRequirement(BackendId backendId) {
        if (backendId == null) {
            throw new NullPointerException("backendId");
        }
        this.backendId = backendId;
    }

    /**
     * Returns the exact backend identity required for later eligible ownership.
     *
     * @return the stored backend identity by the same reference supplied by the caller; never
     *     {@code null}
     */
    public BackendId backendId() {
        return backendId;
    }
}
