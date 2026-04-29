package backend.memory;

/**
 * Reason why execution needs a CPU-visible representation of a runtime tensor.
 *
 * <p>The enum is diagnostic and contractual. It does not perform a transfer by
 * itself; callers that mark a tensor as materialized must already have copied
 * or synchronized the device representation into CPU array storage. Keeping the
 * reason explicit prevents future Metal/GPU-owned paths from silently publishing
 * stale Java arrays.</p>
 */
public enum CpuMaterializationReason {
    /**
     * A later CPU backend step needs to read this value.
     */
    CPU_CONSUMER,

    /**
     * The compiled graph result is being copied back to the user-visible root tensor.
     */
    GRAPH_OUTPUT,

    /**
     * A computed gradient is being published to a user-visible tensor.
     */
    GRADIENT_PUBLICATION,

    /**
     * User-facing tensor data access requested CPU-visible storage.
     */
    PUBLIC_DATA_ACCESS,

    /**
     * An accelerator path fell back to CPU execution and needs CPU storage as the active representation.
     */
    CPU_FALLBACK;

    /**
     * Returns a stable lowercase label suitable for traces and error messages.
     *
     * @return diagnostic label
     */
    public String label() {
        return name().toLowerCase(java.util.Locale.ROOT);
    }
}
