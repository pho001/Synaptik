package graph.execution.trace;

/**
 * Prepare-stage diagnostics.
 *
 * @param measured whether prepare timing was measured
 * @param durationNs prepare duration in nanoseconds
 * @param forwardStepCount number of prepared forward steps
 * @param backwardStepCount number of prepared backward steps
 * @param backendSelection backend selection decisions made during preparation
 */
public record PrepareTrace(
        boolean measured,
        long durationNs,
        int forwardStepCount,
        int backwardStepCount,
        BackendSelectionTrace backendSelection
) {
    public PrepareTrace {
        backendSelection = backendSelection == null ? BackendSelectionTrace.empty() : backendSelection;
    }

    /**
     * Returns a trace marker for skipped or unavailable preparation tracing.
     *
     * @return skipped prepare trace
     */
    public static PrepareTrace skipped() {
        return new PrepareTrace(false, 0L, 0, 0, BackendSelectionTrace.empty());
    }
}
