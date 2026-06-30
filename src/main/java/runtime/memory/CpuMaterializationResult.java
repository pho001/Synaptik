package runtime.memory;

/**
 * Result returned by a device-to-CPU materializer after it has synchronized tensor bytes.
 *
 * <p>The result is diagnostic. The materializer itself is responsible for updating the runtime tensor's
 * CPU-visible storage before returning this value. Execution state uses the result to record trace timing
 * and human-readable detail.</p>
 *
 * @param durationNs measured synchronization duration in nanoseconds
 * @param detail diagnostic detail for traces
 */
public record CpuMaterializationResult(long durationNs, String detail) {
    public CpuMaterializationResult {
        durationNs = Math.max(0L, durationNs);
        detail = detail == null ? "" : detail;
    }

    /**
     * Creates a result with no measured duration.
     *
     * @param detail diagnostic detail
     * @return materialization result
     */
    public static CpuMaterializationResult unmeasured(String detail) {
        return new CpuMaterializationResult(0L, detail);
    }
}
