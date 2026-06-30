package runtime.device.buffer;

import java.util.Objects;

/**
 * Backend-neutral layout ABI v2 support or rejection status.
 */
public record AcceleratorLayoutAbiV2Status(
        AcceleratorLayoutAbiV2StatusCode code,
        String reason,
        int nativeVersion
) {
    public AcceleratorLayoutAbiV2Status {
        code = code == null ? AcceleratorLayoutAbiV2StatusCode.LAYOUT_ABI_UNAVAILABLE : code;
        reason = reason == null ? "" : reason;
        nativeVersion = Math.max(0, nativeVersion);
    }

    public static AcceleratorLayoutAbiV2Status supported(int nativeVersion) {
        return new AcceleratorLayoutAbiV2Status(
                AcceleratorLayoutAbiV2StatusCode.SUPPORTED,
                "",
                nativeVersion
        );
    }

    public static AcceleratorLayoutAbiV2Status unavailable(String reason) {
        return new AcceleratorLayoutAbiV2Status(
                AcceleratorLayoutAbiV2StatusCode.LAYOUT_ABI_UNAVAILABLE,
                reason,
                0
        );
    }

    public boolean supported() {
        return code == AcceleratorLayoutAbiV2StatusCode.SUPPORTED;
    }

    public AcceleratorLayoutAbiV2Status withReason(String detail) {
        return new AcceleratorLayoutAbiV2Status(code, Objects.toString(detail, ""), nativeVersion);
    }
}
