package runtime.state;

import runtime.device.buffer.DeviceBufferBinding;
import runtime.contract.StorageResidency;
import runtime.residency.DeviceBindingRegistry;
import runtime.residency.NativeCpuStorageRegistry;
import runtime.residency.RuntimeResidencyStore;

import java.util.Objects;

/**
 * Run-scoped device buffer bindings and residency transitions.
 */
public final class RuntimeDeviceMemoryState {
    private final RuntimeResidencyStore residencyStore;
    private final DeviceBindingRegistry deviceBindingRegistry;
    private final NativeCpuStorageRegistry nativeStorageRegistry;

    public RuntimeDeviceMemoryState(
            RuntimeResidencyStore residencyStore,
            DeviceBindingRegistry deviceBindingRegistry,
            NativeCpuStorageRegistry nativeStorageRegistry
    ) {
        this.residencyStore = Objects.requireNonNull(residencyStore, "residencyStore cannot be null");
        this.deviceBindingRegistry = Objects.requireNonNull(deviceBindingRegistry, "deviceBindingRegistry cannot be null");
        this.nativeStorageRegistry = Objects.requireNonNull(nativeStorageRegistry, "nativeStorageRegistry cannot be null");
    }

    public void markDeviceCurrent(int nodeId, StorageResidency residency, String deviceBackend, String reason) {
        nativeStorageRegistry.remove(nodeId);
        deviceBindingRegistry.remove(nodeId);
        residencyStore.residencyForNodeId(nodeId).markDeviceCurrent(residency, deviceBackend, reason);
    }

    public void reserveDeviceBufferBinding(int nodeId, DeviceBufferBinding binding) {
        validateDeviceBufferBinding(nodeId, binding);
        deviceBindingRegistry.putReserved(nodeId, binding);
    }

    public void attachDeviceBufferBinding(
            int nodeId,
            DeviceBufferBinding binding,
            StorageResidency residency,
            String reason
    ) {
        Objects.requireNonNull(binding, "binding cannot be null");
        Objects.requireNonNull(residency, "residency cannot be null");
        validateDeviceBufferBinding(nodeId, binding);
        if (residency == StorageResidency.CPU_ARRAY || residency == StorageResidency.CPU_NATIVE) {
            throw new IllegalArgumentException("Device buffer binding requires a device residency.");
        }
        nativeStorageRegistry.remove(nodeId);
        deviceBindingRegistry.putActive(nodeId, binding);
        if (residency == StorageResidency.HOST_SHARED_DEVICE_BUFFER) {
            residencyStore.residencyForNodeId(nodeId).markSharedBufferCurrent(binding.backendId(), reason);
            return;
        }
        residencyStore.residencyForNodeId(nodeId).markDeviceCurrent(residency, binding.backendId(), reason);
    }

    public DeviceBufferBinding deviceBufferBindingForNodeId(int nodeId) {
        residencyStore.residencyForNodeId(nodeId);
        return deviceBindingRegistry.active(nodeId);
    }

    public DeviceBufferBinding writableDeviceBufferBindingForNodeId(int nodeId) {
        residencyStore.residencyForNodeId(nodeId);
        return deviceBindingRegistry.writable(nodeId);
    }

    private void validateDeviceBufferBinding(int nodeId, DeviceBufferBinding binding) {
        Objects.requireNonNull(binding, "binding cannot be null");
        if (binding.nodeId() != nodeId) {
            throw new IllegalArgumentException("Device buffer binding nodeId=" + binding.nodeId()
                    + " does not match requested nodeId=" + nodeId);
        }
        if (!binding.available()) {
            throw new IllegalArgumentException("Device buffer binding is not available: " + binding.describe());
        }
        residencyStore.residencyForNodeId(nodeId);
    }
}
