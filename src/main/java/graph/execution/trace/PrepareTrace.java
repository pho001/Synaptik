package graph.execution.trace;

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

    public static PrepareTrace skipped() {
        return new PrepareTrace(false, 0L, 0, 0, BackendSelectionTrace.empty());
    }
}
