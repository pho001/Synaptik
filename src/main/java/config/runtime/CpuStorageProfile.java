package config.runtime;

/**
 * Runtime-level CPU storage policy.
 *
 * <p>This profile describes whether CPU execution should stay on the Java-array storage path, prefer
 * native {@link java.lang.foreign.MemorySegment}-backed storage for supported operations, or let the
 * runtime planner choose. BLAS storage routes remain kernel-level details inside {@link BlasConfig}.</p>
 */
public enum CpuStorageProfile {
    /**
     * Keep CPU compute on the existing Java-array storage path.
     */
    CPU_ARRAY,

    /**
     * Prefer native CPU storage for supported operations.
     */
    CPU_NATIVE,

    /**
     * Let the runtime planner choose between array and native CPU storage.
     */
    AUTO
}
