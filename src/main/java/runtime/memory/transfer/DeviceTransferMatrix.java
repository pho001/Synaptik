package runtime.memory.transfer;

import runtime.contract.StorageResidency;
import runtime.contract.HostDeviceTransferKind;

/**
 * Backend-neutral transfer matrix for first-wave host/device bridge decisions.
 */
public final class DeviceTransferMatrix {
    private DeviceTransferMatrix() {
    }

    /**
     * Returns the currently implemented support class for a residency transition.
     *
     * <p>Wave 7A only classifies direct native/device routes; actual direct copies are added by backend
     * adapters in the next wave.</p>
     */
    public static DeviceTransferSupport support(StorageResidency source, StorageResidency target) {
        if (source == StorageResidency.CPU_ARRAY
                && (target == StorageResidency.DEVICE_OWNED || target == StorageResidency.HOST_SHARED_DEVICE_BUFFER)) {
            return DeviceTransferSupport.DIRECT;
        }
        if ((source == StorageResidency.DEVICE_OWNED || source == StorageResidency.HOST_SHARED_DEVICE_BUFFER)
                && target == StorageResidency.CPU_ARRAY) {
            return DeviceTransferSupport.DIRECT;
        }
        if (source == StorageResidency.CPU_NATIVE
                && (target == StorageResidency.DEVICE_OWNED || target == StorageResidency.HOST_SHARED_DEVICE_BUFFER)) {
            return DeviceTransferSupport.ARRAY_BRIDGE;
        }
        if ((source == StorageResidency.DEVICE_OWNED || source == StorageResidency.HOST_SHARED_DEVICE_BUFFER)
                && target == StorageResidency.CPU_NATIVE) {
            return DeviceTransferSupport.ARRAY_BRIDGE;
        }
        return source == target ? DeviceTransferSupport.DIRECT : DeviceTransferSupport.UNSUPPORTED;
    }

    public static HostDeviceTransferKind kind(StorageResidency source, StorageResidency target) {
        if (source == StorageResidency.CPU_ARRAY
                && (target == StorageResidency.DEVICE_OWNED || target == StorageResidency.HOST_SHARED_DEVICE_BUFFER)) {
            return HostDeviceTransferKind.CPU_ARRAY_TO_DEVICE_COPY;
        }
        if ((source == StorageResidency.DEVICE_OWNED || source == StorageResidency.HOST_SHARED_DEVICE_BUFFER)
                && target == StorageResidency.CPU_ARRAY) {
            return HostDeviceTransferKind.DEVICE_TO_CPU_ARRAY_COPY;
        }
        if (source == StorageResidency.CPU_NATIVE
                && (target == StorageResidency.DEVICE_OWNED || target == StorageResidency.HOST_SHARED_DEVICE_BUFFER)) {
            return HostDeviceTransferKind.NATIVE_TO_ARRAY_TO_DEVICE_BRIDGE;
        }
        if ((source == StorageResidency.DEVICE_OWNED || source == StorageResidency.HOST_SHARED_DEVICE_BUFFER)
                && target == StorageResidency.CPU_NATIVE) {
            return HostDeviceTransferKind.DEVICE_TO_ARRAY_TO_NATIVE_BRIDGE;
        }
        return HostDeviceTransferKind.SYNC_ONLY;
    }
}
