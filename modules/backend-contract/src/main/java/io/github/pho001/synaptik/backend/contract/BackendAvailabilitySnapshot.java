package io.github.pho001.synaptik.backend.contract;

import java.util.Map;

/**
 * Reports the devices that one backend currently presents as available and the coarse class of
 * each reported device.
 *
 * <p>The snapshot is immutable, caller-supplied point-in-time data. Every device identity must
 * belong to the snapshot's {@link #backendId() backend identity}. The backend identity remains
 * meaningful when the device map is empty; an empty map means that the supplying context reports
 * no currently available device for that backend.</p>
 *
 * <p>Construction validates the source map in its iteration order and then creates an immutable
 * structural copy. The exact backend identity, device identity keys, and class values are
 * retained, but the source map itself is not: later structural changes to a mutable source map do
 * not affect the snapshot. The copied map's iteration order is unspecified and has no contract
 * meaning.</p>
 *
 * <p>This value performs no discovery or registration and provides no capability, ownership,
 * liveness, refresh, preparation, or execution behavior.</p>
 *
 * @param backendId non-null identity that scopes every reported device and remains meaningful for
 *     an empty snapshot; retained by reference
 * @param devices non-null map from same-backend device identities to their coarse classes;
 *     structurally copied, with keys and values retained by reference
 */
public record BackendAvailabilitySnapshot(
        BackendId backendId, Map<BackendDeviceId, DeviceClass> devices) {
    /**
     * Creates an immutable snapshot from caller-supplied availability facts.
     *
     * <p>After validating the two components, each source entry is checked in source-map iteration
     * order. For one entry, its device identity, class, and backend match are checked in that
     * order before validation advances to the next entry.</p>
     *
     * @param backendId non-null identity that scopes every reported device and remains meaningful
     *     for an empty snapshot; retained by reference
     * @param devices non-null map from same-backend device identities to their coarse classes;
     *     structurally copied, with keys and values retained by reference
     * @throws NullPointerException if {@code backendId} is {@code null}, with message
     *     {@code backendId}; if {@code devices} is {@code null}, with message {@code devices}; if
     *     an entry has a null key, with message {@code devices contains null deviceId}; or if an
     *     entry has a null value, with message {@code devices contains null deviceClass}
     * @throws IllegalArgumentException if a device identity belongs to a backend unequal to
     *     {@code backendId}, with message {@code device backendId must match snapshot backendId}
     */
    public BackendAvailabilitySnapshot(
            BackendId backendId, Map<BackendDeviceId, DeviceClass> devices) {
        if (backendId == null) {
            throw new NullPointerException("backendId");
        }
        if (devices == null) {
            throw new NullPointerException("devices");
        }
        for (Map.Entry<BackendDeviceId, DeviceClass> entry : devices.entrySet()) {
            BackendDeviceId deviceId = entry.getKey();
            if (deviceId == null) {
                throw new NullPointerException("devices contains null deviceId");
            }
            DeviceClass deviceClass = entry.getValue();
            if (deviceClass == null) {
                throw new NullPointerException("devices contains null deviceClass");
            }
            if (!backendId.equals(deviceId.backendId())) {
                throw new IllegalArgumentException(
                        "device backendId must match snapshot backendId");
            }
        }
        this.backendId = backendId;
        this.devices = Map.copyOf(devices);
    }

    /**
     * Returns the backend identity that scopes this snapshot.
     *
     * @return the stored backend identity by the same reference supplied by the caller; never
     *     {@code null}
     */
    public BackendId backendId() {
        return backendId;
    }

    /**
     * Returns the immutable structural snapshot of reported device identities and classes.
     *
     * @return the non-null, structurally immutable device-to-class map; its keys and values are
     *     the exact references supplied by the caller, and its iteration order is unspecified
     */
    public Map<BackendDeviceId, DeviceClass> devices() {
        return devices;
    }
}
