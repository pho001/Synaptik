package backend.cpu.nativecpu;

/**
 * Prepared native CPU route selected before execution.
 */
public enum PreparedNativeCpuRoute {
    NONE,
    NATIVE_EXECUTABLE,
    CONDITION_ARRAY_INPUT_NATIVE_OUTPUT,
    VIEW_ALIAS,
    FALLBACK_ONLY
}
