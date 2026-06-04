package backend.cpu1.fused.ir;

public enum Cpu1FusedAccessKind {
    DIRECT_CONTIGUOUS,
    OFFSET_CONTIGUOUS,
    DIRECT_STRIDED,
    OFFSET_STRIDED,
    BROADCAST_STRIDED
}
