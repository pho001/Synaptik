package backend.memory.transfer;

import runtime.contract.StorageResidency;
import runtime.contract.HostDeviceTransferKind;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class DeviceTransferMatrixTest {
    @Test
    void matrixClassifiesExistingArrayDeviceRoutesAsDirectCopies() {
        assertEquals(
                DeviceTransferSupport.DIRECT,
                DeviceTransferMatrix.support(StorageResidency.CPU_ARRAY, StorageResidency.DEVICE_OWNED)
        );
        assertEquals(
                HostDeviceTransferKind.CPU_ARRAY_TO_DEVICE_COPY,
                DeviceTransferMatrix.kind(StorageResidency.CPU_ARRAY, StorageResidency.DEVICE_OWNED)
        );
        assertEquals(
                DeviceTransferSupport.DIRECT,
                DeviceTransferMatrix.support(StorageResidency.DEVICE_OWNED, StorageResidency.CPU_ARRAY)
        );
        assertEquals(
                HostDeviceTransferKind.DEVICE_TO_CPU_ARRAY_COPY,
                DeviceTransferMatrix.kind(StorageResidency.DEVICE_OWNED, StorageResidency.CPU_ARRAY)
        );
    }

    @Test
    void matrixClassifiesNativeDeviceRoutesAsArrayBridgeUntilDirectAdapterExists() {
        assertEquals(
                DeviceTransferSupport.ARRAY_BRIDGE,
                DeviceTransferMatrix.support(StorageResidency.CPU_NATIVE, StorageResidency.DEVICE_OWNED)
        );
        assertEquals(
                HostDeviceTransferKind.NATIVE_TO_ARRAY_TO_DEVICE_BRIDGE,
                DeviceTransferMatrix.kind(StorageResidency.CPU_NATIVE, StorageResidency.DEVICE_OWNED)
        );
        assertEquals(
                DeviceTransferSupport.ARRAY_BRIDGE,
                DeviceTransferMatrix.support(StorageResidency.DEVICE_OWNED, StorageResidency.CPU_NATIVE)
        );
        assertEquals(
                HostDeviceTransferKind.DEVICE_TO_ARRAY_TO_NATIVE_BRIDGE,
                DeviceTransferMatrix.kind(StorageResidency.DEVICE_OWNED, StorageResidency.CPU_NATIVE)
        );
    }
}
