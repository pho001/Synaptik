package backend.accelerator.buffer;

import java.util.Objects;

/**
 * Maps layout ABI v2 status outcomes to stable accelerator buffer reason codes.
 */
public final class AcceleratorLayoutAbiV2ReasonCodes {
    private AcceleratorLayoutAbiV2ReasonCodes() {
    }

    public static AcceleratorBufferReasonCode fromLayoutAbiV2Status(AcceleratorLayoutAbiV2StatusCode statusCode) {
        return switch (Objects.requireNonNull(statusCode, "statusCode cannot be null")) {
            case SUPPORTED -> AcceleratorBufferReasonCode.BUFFER_BINDING_AVAILABLE;
            case LAYOUT_ABI_UNAVAILABLE -> AcceleratorBufferReasonCode.NATIVE_LAYOUT_ABI_UNAVAILABLE;
            case VERSION_MISMATCH -> AcceleratorBufferReasonCode.NATIVE_LAYOUT_ABI_VERSION_MISMATCH;
            case METADATA_UNSUPPORTED -> AcceleratorBufferReasonCode.NATIVE_LAYOUT_METADATA_UNSUPPORTED;
            case RANK_UNSUPPORTED -> AcceleratorBufferReasonCode.NATIVE_LAYOUT_RANK_UNSUPPORTED;
            case DTYPE_UNSUPPORTED -> AcceleratorBufferReasonCode.NATIVE_LAYOUT_DTYPE_UNSUPPORTED;
            case PHYSICAL_SPAN_OVERFLOW -> AcceleratorBufferReasonCode.NATIVE_LAYOUT_PHYSICAL_SPAN_OVERFLOW;
        };
    }
}
