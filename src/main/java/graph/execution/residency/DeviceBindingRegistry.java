package graph.execution.residency;

import backend.memory.DeviceBufferBinding;

import java.util.HashMap;
import java.util.Map;

/**
 * Run-scoped active and reserved device buffer bindings.
 */
public final class DeviceBindingRegistry {
    private final Map<Integer, DeviceBufferBinding> deviceBufferBindingByNodeId;
    private final Map<Integer, DeviceBufferBinding> reservedDeviceBufferBindingByNodeId;

    public DeviceBindingRegistry(Map<Integer, DeviceBufferBinding> deviceBufferBindingByNodeId) {
        this.deviceBufferBindingByNodeId = new HashMap<>(
                deviceBufferBindingByNodeId == null ? Map.of() : deviceBufferBindingByNodeId
        );
        this.reservedDeviceBufferBindingByNodeId = new HashMap<>();
    }

    public void putActive(int nodeId, DeviceBufferBinding binding) {
        deviceBufferBindingByNodeId.put(nodeId, binding);
        reservedDeviceBufferBindingByNodeId.remove(nodeId);
    }

    public void putReserved(int nodeId, DeviceBufferBinding binding) {
        reservedDeviceBufferBindingByNodeId.put(nodeId, binding);
    }

    public DeviceBufferBinding active(int nodeId) {
        return deviceBufferBindingByNodeId.get(nodeId);
    }

    public DeviceBufferBinding writable(int nodeId) {
        DeviceBufferBinding reserved = reservedDeviceBufferBindingByNodeId.get(nodeId);
        return reserved == null ? deviceBufferBindingByNodeId.get(nodeId) : reserved;
    }

    public void remove(int nodeId) {
        deviceBufferBindingByNodeId.remove(nodeId);
        reservedDeviceBufferBindingByNodeId.remove(nodeId);
    }

    public void clear() {
        deviceBufferBindingByNodeId.clear();
        reservedDeviceBufferBindingByNodeId.clear();
    }
}
