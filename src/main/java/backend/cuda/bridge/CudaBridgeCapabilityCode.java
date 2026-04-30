package backend.cuda.bridge;

/**
 * Stable CUDA native bridge capability codes.
 */
public enum CudaBridgeCapabilityCode {
    AVAILABLE,
    NATIVE_LIBRARY_UNAVAILABLE,
    REQUIRED_SYMBOL_MISSING,
    CUDA_RUNTIME_UNAVAILABLE,
    CONTEXT_UNAVAILABLE,
    GRAPH_EXECUTION_ABI_UNAVAILABLE,
    BUFFER_ABI_UNAVAILABLE
}
