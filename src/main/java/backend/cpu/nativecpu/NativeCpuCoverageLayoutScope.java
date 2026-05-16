package backend.cpu.nativecpu;

/**
 * Layout scope described by the native CPU coverage matrix.
 */
public enum NativeCpuCoverageLayoutScope {
    DENSE_CONTIGUOUS,
    VIEW_ONLY,
    STRIDED_UNSUPPORTED,
    ARRAY_ONLY
}
