package backend.cpu.fused.ir;

/**
 * Internal fused-codegen classification for how an external input is indexed.
 */
public enum FusedAccessKind {
    DIRECT_CONTIGUOUS,
    DIRECT_STRIDED,
    OFFSET_CONTIGUOUS,
    BROADCAST_STRIDED,
    OFFSET_STRIDED
}
