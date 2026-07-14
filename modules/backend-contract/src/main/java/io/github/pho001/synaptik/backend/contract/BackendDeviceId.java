package io.github.pho001.synaptik.backend.contract;

/**
 * Identifies one device within an owning backend's opaque device-token namespace.
 *
 * <p>The {@link BackendId} scopes the device token, so equal tokens from different backend
 * domains identify different devices. The exact backend identity and nonblank {@link String}
 * references supplied by the caller are retained without normalization. Equality, hashing, and
 * diagnostic text follow ordinary record semantics over both components in component order.</p>
 *
 * <p>This immutable descriptive value does not prove that the backend or device is registered,
 * discovered, present, available, capable, or accessible. It contains no device handle or live
 * service and performs no resource, preparation, or execution work.</p>
 *
 * @param backendId non-null identity that owns and scopes the device-token namespace; retained by
 *     reference
 * @param value opaque nonblank backend-defined device token to retain by reference without
 *     normalization; must not be {@code null}
 */
public record BackendDeviceId(BackendId backendId, String value) {
    /**
     * Creates a backend-scoped device identity from exact caller-supplied components.
     *
     * <p>Validation follows component order: {@code backendId} is validated before {@code value}.
     * No component is copied or normalized.</p>
     *
     * @param backendId non-null identity that owns and scopes the device-token namespace; retained
     *     by reference
     * @param value opaque nonblank backend-defined device token to retain by reference without
     *     normalization; must not be {@code null}
     * @throws NullPointerException if {@code backendId} is {@code null}, with message
     *     {@code backendId}, or if {@code value} is {@code null} after a valid backend identity,
     *     with message {@code value}
     * @throws IllegalArgumentException if {@code value} is blank according to
     *     {@link String#isBlank()}; the exception message is {@code value must not be blank}
     */
    public BackendDeviceId(BackendId backendId, String value) {
        if (backendId == null) {
            throw new NullPointerException("backendId");
        }
        if (value == null) {
            throw new NullPointerException("value");
        }
        if (value.isBlank()) {
            throw new IllegalArgumentException("value must not be blank");
        }
        this.backendId = backendId;
        this.value = value;
    }

    /**
     * Returns the backend identity that scopes the device token.
     *
     * @return the stored backend identity by the same reference supplied by the caller; never
     *     {@code null}
     */
    public BackendId backendId() {
        return backendId;
    }

    /**
     * Returns the exact opaque device token supplied at construction.
     *
     * @return the stored nonblank token by the same reference supplied by the caller; never
     *     {@code null}
     */
    public String value() {
        return value;
    }
}
