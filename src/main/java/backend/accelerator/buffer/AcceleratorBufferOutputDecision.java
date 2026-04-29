package backend.accelerator.buffer;

/**
 * Per-output diagnostics for accelerator buffer preflight.
 *
 * @param nodeId semantic output node id
 * @param accepted whether the output can be represented as a backend buffer
 * @param reasonCode stable reason code
 * @param reason human-readable detail
 */
public record AcceleratorBufferOutputDecision(
        int nodeId,
        boolean accepted,
        AcceleratorBufferReasonCode reasonCode,
        String reason
) {
    public AcceleratorBufferOutputDecision {
        reasonCode = reasonCode == null ? AcceleratorBufferReasonCode.NOT_EVALUATED : reasonCode;
        reason = reason == null ? "" : reason;
    }
}
