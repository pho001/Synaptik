package backend.cpu.nativecpu;

/**
 * Native CPU coverage and performance classification for a semantic operation.
 */
public enum NativeCpuKernelPerformanceStatus {
    NATIVE_FAST,
    NATIVE_CORRECT_BUT_SLOW,
    NATIVE_UNSUPPORTED,
    ARRAY_ONLY,
    VIEW_ONLY,
    LIBRARY_PROVIDER
}
