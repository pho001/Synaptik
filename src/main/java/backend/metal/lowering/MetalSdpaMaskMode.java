package backend.metal.lowering;

/**
 * Compile-time mask semantic modes for direct Metal SDPA planning.
 */
enum MetalSdpaMaskMode {
    UNMASKED,
    EXTERNAL_BOOL_MASK,
    CAUSAL_BOOL_MASK,
    EXTERNAL_AND_CAUSAL_BOOL_MASK,
    INVALID
}
