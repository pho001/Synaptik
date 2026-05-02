package backend.metal.bridge;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class MetalMpsBridgeExecutionStatsTest {
    @Test
    void fallbackStatsDeclareCpuFallbackPath() {
        MetalMpsBridgeExecutionStats stats = MetalMpsBridgeExecutionStats.fallback(
                "bridge unavailable",
                2,
                1,
                128L,
                64L
        );

        assertTrue(stats.usedCpuFallback());
        assertEquals("bridge unavailable", stats.fallbackReason());
        assertEquals(MetalMpsBridgeExecutionPath.CPU_FALLBACK, stats.executionPath());
        assertEquals(MetalNativeCopyStrategy.UNKNOWN_OR_UNPROVEN, stats.nativeCopyStrategy());
        assertEquals(128L, stats.inputBytes());
        assertEquals(64L, stats.outputBytes());
        assertEquals(0L, stats.nativeDeviceCopyNs());
        assertTrue(!stats.outputBufferWriteProven());
    }

    @Test
    void missingPathDefaultsToTensorArrayCopyForNonFallbackStats() {
        MetalMpsBridgeExecutionStats stats = new MetalMpsBridgeExecutionStats(
                false,
                null,
                null,
                1,
                1,
                4L,
                4L,
                10L,
                20L,
                30L,
                40L,
                50L,
                100L
        );

        assertEquals("", stats.fallbackReason());
        assertEquals(MetalMpsBridgeExecutionPath.TENSOR_ARRAY_COPY, stats.executionPath());
        assertEquals(MetalNativeCopyStrategy.MPSGRAPH_RESULT_COPY, stats.nativeCopyStrategy());
        assertEquals(40L, stats.nativeDeviceCopyNs());
        assertEquals(50L, stats.nativeToJavaCopyNs());
    }

    @Test
    void trueOutputBufferWriteStrategyMarksProofFlag() {
        MetalMpsBridgeExecutionStats stats = new MetalMpsBridgeExecutionStats(
                false,
                "",
                MetalMpsBridgeExecutionPath.BUFFER_BINDING,
                MetalNativeCopyStrategy.TRUE_OUTPUT_BUFFER_WRITE,
                1,
                1,
                4L,
                4L,
                0L,
                0L,
                30L,
                0L,
                0L,
                30L
        );

        assertEquals(MetalNativeCopyStrategy.TRUE_OUTPUT_BUFFER_WRITE, stats.nativeCopyStrategy());
        assertTrue(stats.outputBufferWriteProven());
    }
}
