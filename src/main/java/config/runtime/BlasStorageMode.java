package config.runtime;

/**
 * Storage route policy for BLAS-backed CPU matmul dispatch.
 */
public enum BlasStorageMode {
    /**
     * Use the existing Java-array OpenBLAS bridge, which copies arrays into native call buffers.
     */
    CPU_ARRAY,

    /**
     * Use MemorySegment-backed native CPU storage when the dtype/layout is supported.
     */
    CPU_NATIVE,

    /**
     * Let the planner choose a native segment route for supported large dense F32/F64 GEMM.
     */
    AUTO
}
