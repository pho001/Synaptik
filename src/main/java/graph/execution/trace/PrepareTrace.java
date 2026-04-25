package graph.execution.trace;

public record PrepareTrace(
        boolean measured,
        long durationNs,
        int forwardStepCount,
        int backwardStepCount,
        AcceleratorSelectionTrace acceleratorSelection
) {
    public PrepareTrace {
        acceleratorSelection = acceleratorSelection == null ? AcceleratorSelectionTrace.empty() : acceleratorSelection;
    }

    public static PrepareTrace skipped() {
        return new PrepareTrace(false, 0L, 0, 0, AcceleratorSelectionTrace.empty());
    }
}
