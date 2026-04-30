package backend.cuda.bridge;

/**
 * Layered CUDA bridge capability state.
 *
 * @param nativeLibraryAvailable whether a native library lookup was resolved
 * @param cudaRuntimeAvailable whether the shim reports CUDA runtime/device availability
 * @param contextAvailable whether a CUDA context can be created in principle
 * @param graphExecutionAvailable whether graph compile/execute ABI symbols are present
 * @param bufferExecutionSupported whether native buffer execution can be used
 * @param layoutAbiV2Supported whether layout ABI v2 metadata symbols can be used
 * @param layoutAbiV2Version native layout ABI version, or 0 when unavailable
 * @param code stable capability code
 * @param reason human-readable diagnostic reason
 */
public record CudaBridgeCapabilities(
        boolean nativeLibraryAvailable,
        boolean cudaRuntimeAvailable,
        boolean contextAvailable,
        boolean graphExecutionAvailable,
        boolean bufferExecutionSupported,
        boolean layoutAbiV2Supported,
        int layoutAbiV2Version,
        CudaBridgeCapabilityCode code,
        String reason
) {
    public CudaBridgeCapabilities {
        layoutAbiV2Version = Math.max(0, layoutAbiV2Version);
        code = code == null ? CudaBridgeCapabilityCode.NATIVE_LIBRARY_UNAVAILABLE : code;
        reason = reason == null ? "" : reason;
    }

    /**
     * Returns fully available graph capability state.
     */
    public static CudaBridgeCapabilities available(boolean bufferExecutionSupported) {
        return available(bufferExecutionSupported, false, 0);
    }

    /**
     * Returns fully available graph capability state with layout ABI detail.
     */
    public static CudaBridgeCapabilities available(
            boolean bufferExecutionSupported,
            boolean layoutAbiV2Supported,
            int layoutAbiV2Version
    ) {
        return new CudaBridgeCapabilities(
                true,
                true,
                true,
                true,
                bufferExecutionSupported,
                layoutAbiV2Supported,
                layoutAbiV2Version,
                CudaBridgeCapabilityCode.AVAILABLE,
                ""
        );
    }

    /**
     * Returns unavailable capability state with a stable code.
     */
    public static CudaBridgeCapabilities unavailable(CudaBridgeCapabilityCode code, String reason) {
        return new CudaBridgeCapabilities(false, false, false, false, false, false, 0, code, reason);
    }
}
