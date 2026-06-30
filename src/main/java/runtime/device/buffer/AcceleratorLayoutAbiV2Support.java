package runtime.device.buffer;

/**
 * Shared constants and helpers for layout ABI v2 capability checks.
 */
public final class AcceleratorLayoutAbiV2Support {
    public static final int REQUIRED_VERSION = 2;

    private AcceleratorLayoutAbiV2Support() {
    }

    public static AcceleratorLayoutAbiV2Status fromNativeVersion(int nativeVersion, String unavailableReason) {
        if (nativeVersion <= 0) {
            return AcceleratorLayoutAbiV2Status.unavailable(unavailableReason);
        }
        if (nativeVersion != REQUIRED_VERSION) {
            return new AcceleratorLayoutAbiV2Status(
                    AcceleratorLayoutAbiV2StatusCode.VERSION_MISMATCH,
                    "layout ABI v2 version mismatch: expected " + REQUIRED_VERSION + ", got " + nativeVersion,
                    nativeVersion
            );
        }
        return AcceleratorLayoutAbiV2Status.supported(nativeVersion);
    }
}
