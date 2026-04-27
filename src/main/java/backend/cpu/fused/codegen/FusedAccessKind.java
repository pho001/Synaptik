package backend.cpu.fused.codegen;

public enum FusedAccessKind {
    DIRECT_CONTIGUOUS,
    DIRECT_STRIDED,
    OFFSET_CONTIGUOUS,
    BROADCAST_STRIDED,
    OFFSET_STRIDED
}
