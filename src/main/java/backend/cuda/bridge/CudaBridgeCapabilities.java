package backend.cuda.bridge;

/**
 * Layered CUDA bridge capability state.
 *
 * @param nativeLibraryAvailable whether a native library lookup was resolved
 * @param cudaRuntimeAvailable whether the shim reports CUDA runtime/device availability
 * @param contextAvailable whether a CUDA context can be created in principle
 * @param graphExecutionAvailable whether graph compile/execute ABI symbols are present
 * @param bufferExecutionSupported whether native buffer execution can be used
 * @param code stable capability code
 * @param reason human-readable diagnostic reason
 */
public record CudaBridgeCapabilities(
        boolean nativeLibraryAvailable,
        boolean cudaRuntimeAvailable,
        boolean contextAvailable,
        boolean graphExecutionAvailable,
        boolean bufferExecutionSupported,
        CudaBridgeCapabilityCode code,
        String reason
) {
    public CudaBridgeCapabilities {
        code = code == null ? CudaBridgeCapabilityCode.NATIVE_LIBRARY_UNAVAILABLE : code;
        reason = reason == null ? "" : reason;
    }

    /**
     * Returns fully available graph capability state.
     */
    public static CudaBridgeCapabilities available(boolean bufferExecutionSupported) {
        return new CudaBridgeCapabilities(
                true,
                true,
                true,
                true,
                bufferExecutionSupported,
                CudaBridgeCapabilityCode.AVAILABLE,
                ""
        );
    }

    /**
     * Returns unavailable capability state with a stable code.
     */
    public static CudaBridgeCapabilities unavailable(CudaBridgeCapabilityCode code, String reason) {
        return new CudaBridgeCapabilities(false, false, false, false, false, code, reason);
    }
}
