package backend.accelerator.buffer;

/**
 * Stable backend-neutral categories for logical tensor layouts held by accelerator buffers.
 */
public enum AcceleratorBufferLayoutClass {
    DENSE_CONTIGUOUS,
    ZERO_OFFSET_VIEW,
    NON_ZERO_OFFSET_VIEW,
    PERMUTED_OR_STRIDED_VIEW,
    BROADCAST_ZERO_STRIDE_VIEW,
    UNSUPPORTED
}
