package backend.accelerator.buffer;

import java.util.Objects;

/**
 * Per-output diagnostics for accelerator buffer preflight.
 *
 * @param nodeId semantic output node id
 * @param layout runtime/request layout evaluated for this output
 * @param accepted whether the output can be represented as a backend buffer
 * @param reasonCode stable reason code
 * @param reason human-readable detail
 */
public record AcceleratorBufferOutputDecision(
        int nodeId,
        AcceleratorBufferLayout layout,
        boolean accepted,
        AcceleratorBufferReasonCode reasonCode,
        String reason
) {
    public AcceleratorBufferOutputDecision {
        Objects.requireNonNull(layout, "layout cannot be null");
        reasonCode = reasonCode == null ? AcceleratorBufferReasonCode.NOT_EVALUATED : reasonCode;
        reason = reason == null ? "" : reason;
    }
}
