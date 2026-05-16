package backend.cpu.nativecpu;

/**
 * Input residency policy implied by a prepared native CPU route.
 */
public enum PreparedNativeCpuInputPolicy {
    ALL_CPU,
    ALL_NATIVE,
    CONDITION_CPU_VALUES_NATIVE
}
