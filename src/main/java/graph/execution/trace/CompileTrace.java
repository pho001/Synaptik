package graph.execution.trace;

/**
 * Compile-stage diagnostics.
 *
 * @param measured whether compile timing was measured
 * @param durationNs compile duration in nanoseconds
 * @param totalNodeCount total nodes in the final compiled graph
 * @param forwardNodeCount nodes in the forward portion of the graph
 * @param supportsBackward whether backward artifacts were compiled
 * @param partitionPlanning partition planning trace captured during compile
 */
public record CompileTrace(
        boolean measured,
        long durationNs,
        int totalNodeCount,
        int forwardNodeCount,
        boolean supportsBackward,
        PartitionCompileTrace partitionPlanning
) {
    /**
     * Returns a trace marker for skipped or unavailable compile tracing.
     *
     * @return skipped compile trace
     */
    public static CompileTrace skipped() {
        return new CompileTrace(false, 0L, 0, 0, false, PartitionCompileTrace.empty());
    }
}
