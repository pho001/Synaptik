package backend.metal.bridge;

/**
 * Stable Metal native bridge capability codes.
 */
public enum MetalMpsCapabilityCode {
    AVAILABLE,
    NATIVE_LIBRARY_UNAVAILABLE,
    REQUIRED_SYMBOL_MISSING,
    RUNTIME_UNAVAILABLE,
    GRAPH_EXECUTION_ABI_UNAVAILABLE,
    BUFFER_ABI_UNAVAILABLE,
    LAYOUT_ABI_V2_UNAVAILABLE,
    LAYOUT_ABI_V2_VERSION_MISMATCH
}
