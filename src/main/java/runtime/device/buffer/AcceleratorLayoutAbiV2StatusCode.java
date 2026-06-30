package runtime.device.buffer;

/**
 * Stable metadata validation outcomes for layout ABI v2.
 */
public enum AcceleratorLayoutAbiV2StatusCode {
    SUPPORTED,
    LAYOUT_ABI_UNAVAILABLE,
    VERSION_MISMATCH,
    METADATA_UNSUPPORTED,
    RANK_UNSUPPORTED,
    DTYPE_UNSUPPORTED,
    PHYSICAL_SPAN_OVERFLOW
}
