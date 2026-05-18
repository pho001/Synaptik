package backend.cpu.nativecpu;

/**
 * Executable storage/compute paths tracked by the native CPU parity matrix.
 */
public enum NativeCpuStoragePath {
    CPU_ARRAY_DENSE,
    CPU_ARRAY_STRIDED,
    CPU_NATIVE_SINGLE_DENSE,
    CPU_NATIVE_REGION_DENSE,
    CPU_NATIVE_REGION_STRIDED,
    CPU_NATIVE_REGION_BROADCAST,
    CPU_NATIVE_REGION_VIEW_ALIAS,
    CPU_NATIVE_REGION_PROVIDER
}
