package backend.cpu.nativecpu.layout;

/**
 * Physical MemorySegment kernel family for CPU-native layout execution.
 */
public enum NativeSegmentKernelFamily {
    SEGMENT_DENSE_SCALAR,
    SEGMENT_STRIDED_SCALAR,
    SEGMENT_PARALLEL,
    SEGMENT_VECTOR,
    SEGMENT_FUSED,
    PROVIDER,
    VIEW_ALIAS
}
