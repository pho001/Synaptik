package backend.memory;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class TensorResidencyStateTest {
    @Test
    void deviceWriteMakesCpuMaterializationRequired() {
        TensorResidencyState state = TensorResidencyState.cpuArrayCurrent("input");

        state.markDeviceCurrent(StorageResidency.DEVICE_OWNED, "GPU_METAL", "metal output");

        assertEquals(StorageResidency.DEVICE_OWNED, state.residency());
        assertFalse(state.cpuCurrent());
        assertTrue(state.deviceCurrent());
        assertTrue(state.requiresCpuMaterialization());
        assertEquals("GPU_METAL", state.deviceBackend());
        assertEquals("metal output", state.lastTransitionReason());
    }

    @Test
    void materializationReturnsToCpuArrayResidency() {
        TensorResidencyState state = TensorResidencyState.cpuArrayStale("allocated");
        state.markDeviceCurrent(StorageResidency.HOST_SHARED_DEVICE_BUFFER, "GPU_METAL", "shared output");

        state.markMaterializedToCpu("public read");

        assertEquals(StorageResidency.CPU_ARRAY, state.residency());
        assertTrue(state.cpuCurrent());
        assertFalse(state.deviceCurrent());
        assertFalse(state.requiresCpuMaterialization());
        assertEquals("", state.deviceBackend());
        assertEquals("public read", state.lastTransitionReason());
    }

    @Test
    void nativeWriteMakesCpuMaterializationRequiredWithoutDeviceResidency() {
        TensorResidencyState state = TensorResidencyState.cpuArrayCurrent("input");

        state.markNativeCurrent("native output");

        assertEquals(StorageResidency.CPU_NATIVE, state.residency());
        assertFalse(state.cpuCurrent());
        assertTrue(state.nativeCurrent());
        assertFalse(state.deviceCurrent());
        assertTrue(state.requiresCpuMaterialization());
        assertEquals("", state.deviceBackend());
        assertEquals("native output", state.lastTransitionReason());
    }

    @Test
    void sharedBufferCanBeCurrentForCpuAndDevice() {
        TensorResidencyState state = TensorResidencyState.cpuArrayCurrent("input");

        state.markSharedBufferCurrent("GPU_METAL", "shared buffer attached");

        assertEquals(StorageResidency.HOST_SHARED_DEVICE_BUFFER, state.residency());
        assertTrue(state.cpuCurrent());
        assertTrue(state.deviceCurrent());
        assertFalse(state.requiresCpuMaterialization());
        assertEquals("GPU_METAL", state.deviceBackend());
        assertEquals("shared buffer attached", state.lastTransitionReason());
    }

    @Test
    void deviceWriteRejectsCpuArrayResidency() {
        TensorResidencyState state = TensorResidencyState.cpuArrayCurrent("input");

        assertThrows(
                IllegalArgumentException.class,
                () -> state.markDeviceCurrent(StorageResidency.CPU_ARRAY, "GPU_METAL", "invalid")
        );
    }

    @Test
    void deviceWriteRejectsCpuNativeResidency() {
        TensorResidencyState state = TensorResidencyState.cpuArrayCurrent("input");

        assertThrows(
                IllegalArgumentException.class,
                () -> state.markDeviceCurrent(StorageResidency.CPU_NATIVE, "GPU_METAL", "invalid")
        );
    }
}
