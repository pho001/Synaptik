package backend.cpu.nativecpu;

/**
 * Result storage residency exposed by the native CPU parity matrix.
 */
public enum NativeCpuResultResidency {
    CPU_ARRAY,
    CPU_NATIVE,
    BOOL_MASK_ARRAY,
    BOOL_MASK_NATIVE,
    VIEW_ALIAS
}
