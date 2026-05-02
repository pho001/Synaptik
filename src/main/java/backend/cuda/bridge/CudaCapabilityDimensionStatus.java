package backend.cuda.bridge;

/**
 * Stable status for one CUDA capability dimension.
 */
public enum CudaCapabilityDimensionStatus {
    AVAILABLE,
    UNAVAILABLE,
    VERSION_MISMATCH,
    NOT_INTEGRATED,
    UNKNOWN
}
