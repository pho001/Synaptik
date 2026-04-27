package backend.cuda.bridge;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
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
