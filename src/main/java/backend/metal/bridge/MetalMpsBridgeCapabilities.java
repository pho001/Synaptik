package backend.metal.bridge;

/**
 * Layered Metal MPS bridge capability state.
 *
 * @param nativeLibraryAvailable whether a native library lookup was resolved
 * @param runtimeAvailable whether the Metal/MPS runtime can be used in principle
 * @param contextAvailable whether a Metal context can be created in principle
 * @param graphExecutionAvailable whether graph compile/execute ABI symbols are present
 * @param bufferExecutionSupported whether native buffer execution can be used
 * @param layoutAbiV2Supported whether layout ABI v2 metadata symbols can be used
 * @param layoutAbiV2Version native layout ABI version, or 0 when unavailable
 * @param code stable capability code
 * @param reason human-readable diagnostic reason
 */
public record MetalMpsBridgeCapabilities(
        boolean nativeLibraryAvailable,
        boolean runtimeAvailable,
        boolean contextAvailable,
        boolean graphExecutionAvailable,
        boolean bufferExecutionSupported,
        boolean layoutAbiV2Supported,
        int layoutAbiV2Version,
        MetalMpsCapabilityCode code,
        String reason
) {
    public MetalMpsBridgeCapabilities {
        layoutAbiV2Version = Math.max(0, layoutAbiV2Version);
        code = code == null ? MetalMpsCapabilityCode.NATIVE_LIBRARY_UNAVAILABLE : code;
        reason = reason == null ? "" : reason;
    }

    public static MetalMpsBridgeCapabilities unavailable(MetalMpsCapabilityCode code, String reason) {
        return new MetalMpsBridgeCapabilities(
                false,
                false,
                false,
                false,
                false,
                false,
                0,
                code,
                reason
        );
    }
}
