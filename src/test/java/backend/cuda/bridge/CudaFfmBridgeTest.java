package backend.cuda.bridge;

import backend.accelerator.buffer.AcceleratorLayoutAbiV2Support;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

class CudaFfmBridgeTest {
    @Test
    void bridgeReportsAvailabilityAndProducesContextWithoutThrowing() {
        CudaFfmBridge bridge = new CudaFfmBridge();

        assertNotNull(bridge.unavailableReason());
        CudaBridgeContext bridgeContext = bridge.createContext();
        assertNotNull(bridgeContext);
        if (!bridge.isAvailable()) {
            assertFalse(bridge.unavailableReason().isBlank());
            assertFalse(bridgeContext.available());
        } else {
            assertTrue(bridgeContext.available());
        }
        assertFalse(bridge.supportsBufferBindings());
    }

    @Test
    void capabilitiesReportNativeAndBufferStateWithoutThrowing() {
        CudaFfmBridge bridge = new CudaFfmBridge();

        CudaBridgeCapabilities capabilities = bridge.capabilities();

        assertNotNull(capabilities);
        assertNotNull(capabilities.reason());
        assertEquals(bridge.supportsBufferBindings(), capabilities.bufferExecutionSupported());
        assertTrue(capabilities.layoutAbiV2Version() >= 0);
        if (capabilities.layoutAbiV2Version() < AcceleratorLayoutAbiV2Support.REQUIRED_VERSION) {
            assertFalse(capabilities.layoutAbiV2Supported());
        }
        if (!bridge.isAvailable()) {
            assertNotEquals(CudaBridgeCapabilityCode.AVAILABLE, capabilities.code());
            assertFalse(bridge.createContext().available());
        }
    }

    @Test
    void capabilitiesReportLayoutAbiV2StateWithoutThrowing() {
        CudaFfmBridge bridge = new CudaFfmBridge();

        CudaBridgeCapabilities capabilities = bridge.capabilities();

        assertNotNull(capabilities);
        assertNotNull(capabilities.reason());
        assertTrue(capabilities.layoutAbiV2Version() >= 0);
        if (capabilities.layoutAbiV2Version() < AcceleratorLayoutAbiV2Support.REQUIRED_VERSION) {
            assertFalse(capabilities.layoutAbiV2Supported());
        }
    }

    @Test
    void unconfiguredBridgeUnavailableReasonIsNonBlankWhenUnavailable() {
        CudaFfmBridge bridge = new CudaFfmBridge();

        if (!bridge.isAvailable()) {
            assertFalse(bridge.unavailableReason().isBlank());
            assertFalse(bridge.capabilities().reason().isBlank());
        }
    }

    @Test
    void explicitShimLibraryLoadsWhenConfigured() {
        String explicitLib = System.getProperty("synaptik.cuda.graph.lib");
        assumeTrue(explicitLib != null && !explicitLib.isBlank());

        CudaFfmBridge bridge = new CudaFfmBridge();

        assertNotNull(bridge.unavailableReason());
        if (bridge.isAvailable()) {
            assertTrue(bridge.createContext().available());
        } else {
            assertFalse(bridge.unavailableReason().isBlank());
        }
    }
}
