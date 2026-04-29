package backend.accelerator.buffer;

/**
 * Per-input diagnostics for accelerator buffer preflight.
 *
 * @param nodeId semantic external input node id
 * @param preparedInputUsed whether execution uses a prepared/remapped input tensor
 * @param accepted whether the input can be represented as a backend buffer
 * @param reasonCode stable reason code
 * @param reason human-readable detail
 */
public record AcceleratorBufferInputDecision(
        int nodeId,
        boolean preparedInputUsed,
        boolean accepted,
        AcceleratorBufferReasonCode reasonCode,
        String reason
) {
    public AcceleratorBufferInputDecision {
        reasonCode = reasonCode == null ? AcceleratorBufferReasonCode.NOT_EVALUATED : reasonCode;
        reason = reason == null ? "" : reason;
    }
}
