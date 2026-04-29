package backend.accelerator.buffer;

import java.util.Objects;

/**
 * Per-input diagnostics for accelerator buffer preflight.
 *
 * @param nodeId semantic external input node id
 * @param layout runtime layout evaluated for this input
 * @param preparedInputUsed whether execution uses a prepared/remapped input tensor
 * @param accepted whether the input can be represented as a backend buffer
 * @param reasonCode stable reason code
 * @param reason human-readable detail
 */
public record AcceleratorBufferInputDecision(
        int nodeId,
        AcceleratorBufferLayout layout,
        boolean preparedInputUsed,
        boolean accepted,
        AcceleratorBufferReasonCode reasonCode,
        String reason
) {
    public AcceleratorBufferInputDecision {
        Objects.requireNonNull(layout, "layout cannot be null");
        reasonCode = reasonCode == null ? AcceleratorBufferReasonCode.NOT_EVALUATED : reasonCode;
        reason = reason == null ? "" : reason;
    }
}
