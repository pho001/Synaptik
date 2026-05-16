package backend.cpu.nativecpu;

/**
 * Physical native CPU execution family used by planner facts.
 */
public enum NativeCpuKernelFamily {
    OPENBLAS_NATIVE_SEGMENT,
    SEGMENT_SCALAR,
    VECTOR_API,
    GENERATED_DIRECT,
    NATIVE_MICROKERNEL,
    VIEW_ONLY,
    ARRAY_ONLY
}
