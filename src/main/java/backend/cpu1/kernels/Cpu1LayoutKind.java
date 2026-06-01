package backend.cpu1.kernels;

/**
 * Tensor layout family supported by a cpu1 kernel variant.
 */
public enum Cpu1LayoutKind {
    CONTIGUOUS,
    BROADCAST_INNER,
    STRIDED_RANK2,
    STRIDED_RANK3,
    STRIDED_RANK4,
    STRIDED_GENERIC
}
