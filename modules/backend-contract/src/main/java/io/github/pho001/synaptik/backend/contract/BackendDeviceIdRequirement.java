package io.github.pho001.synaptik.backend.contract;

/**
 * Requires one exact backend-scoped device identity as a later eligible target.
 *
 * <p>Later eligibility targets a {@link BackendDeviceId} equal to the stored identity. That
 * identity also fixes its owning backend, so no separate backend component is needed. The exact
 * non-null reference supplied by the caller is retained. Equality, hashing, and diagnostic text
 * follow ordinary record semantics over that identity.</p>
 *
 * <p>This immutable target does not prove that the device is registered, available, capable, or
 * preparable and does not evaluate eligibility.</p>
 *
 * @param deviceId non-null exact backend-scoped device identity required as a later eligible
 *     target; retained by reference
 */
public record BackendDeviceIdRequirement(BackendDeviceId deviceId) implements BackendRequirement {
    /**
     * Creates an exact-device hard-eligibility target.
     *
     * @param deviceId non-null exact backend-scoped device identity required as a later eligible
     *     target; retained by reference
     * @throws NullPointerException if {@code deviceId} is {@code null}; the exception message is
     *     {@code deviceId}
     */
    public BackendDeviceIdRequirement(BackendDeviceId deviceId) {
        if (deviceId == null) {
            throw new NullPointerException("deviceId");
        }
        this.deviceId = deviceId;
    }

    /**
     * Returns the exact backend-scoped device identity required as a later eligible target.
     *
     * @return the stored device identity by the same reference supplied by the caller; never
     *     {@code null}
     */
    public BackendDeviceId deviceId() {
        return deviceId;
    }
}
