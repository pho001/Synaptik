package io.github.pho001.synaptik.backend.contract;

/**
 * Requires any later eligible device in one coarse device class.
 *
 * <p>The target does not identify one backend or device: any later eligible device whose class is
 * the stored {@link DeviceClass} can satisfy it. The exact non-null enum reference supplied by the
 * caller is retained. Equality, hashing, and diagnostic text follow ordinary record semantics
 * over that class.</p>
 *
 * <p>This immutable target does not prove that a device of the class is registered, available,
 * capable, or preparable and does not evaluate eligibility.</p>
 *
 * @param deviceClass non-null coarse class required of a later eligible device; retained by
 *     reference
 */
public record DeviceClassRequirement(DeviceClass deviceClass) implements BackendRequirement {
    /**
     * Creates a coarse device-class hard-eligibility target.
     *
     * @param deviceClass non-null coarse class required of a later eligible device; retained by
     *     reference
     * @throws NullPointerException if {@code deviceClass} is {@code null}; the exception message
     *     is {@code deviceClass}
     */
    public DeviceClassRequirement(DeviceClass deviceClass) {
        if (deviceClass == null) {
            throw new NullPointerException("deviceClass");
        }
        this.deviceClass = deviceClass;
    }

    /**
     * Returns the coarse class required of a later eligible device.
     *
     * @return the stored device class by the same enum reference supplied by the caller; never
     *     {@code null}
     */
    public DeviceClass deviceClass() {
        return deviceClass;
    }
}
