package backend.metal.bridge;

import java.util.Objects;

/**
 * Diagnostic evidence for whether MPSGraph writes directly into caller-supplied output buffers.
 *
 * @param status proof classification
 * @param detail concise diagnostic detail
 * @param copiedExecutionStats reference execution stats using the conservative result-copy path
 * @param probeExecutionStats no-copy probe execution stats
 */
public record MetalOutputBufferWriteProbeResult(
        MetalOutputBufferWriteProbeStatus status,
        String detail,
        MetalMpsBridgeExecutionStats copiedExecutionStats,
        MetalMpsBridgeExecutionStats probeExecutionStats
) {
    public MetalOutputBufferWriteProbeResult {
        status = Objects.requireNonNullElse(status, MetalOutputBufferWriteProbeStatus.UNSUPPORTED);
        detail = detail == null ? "" : detail;
    }

    /**
     * Returns whether this probe proves that caller-supplied output buffers contain the same bytes as
     * the conservative copied MPSGraph result.
     */
    public boolean provenTrueOutputBufferWrite() {
        return status == MetalOutputBufferWriteProbeStatus.MATCHES_COPIED_RESULT;
    }

    public static MetalOutputBufferWriteProbeResult unsupported(String detail) {
        return new MetalOutputBufferWriteProbeResult(
                MetalOutputBufferWriteProbeStatus.UNSUPPORTED,
                detail,
                null,
                null
        );
    }
}
