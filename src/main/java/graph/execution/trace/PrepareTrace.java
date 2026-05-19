package graph.execution.trace;

import java.util.List;

/**
 * Prepare-stage diagnostics.
 *
 * @param measured whether prepare timing was measured
 * @param durationNs prepare duration in nanoseconds
 * @param forwardStepCount number of prepared forward steps
 * @param backwardStepCount number of prepared backward steps
 * @param backendSelection backend selection decisions made during preparation
 * @param backendDiagnostics backend contributor diagnostics captured during preparation
 */
public record PrepareTrace(
        boolean measured,
        long durationNs,
        int forwardStepCount,
        int backwardStepCount,
        BackendSelectionTrace backendSelection,
        List<BackendPrepareDiagnosticTrace> backendDiagnostics
) {
    public PrepareTrace {
        backendSelection = backendSelection == null ? BackendSelectionTrace.empty() : backendSelection;
        backendDiagnostics = List.copyOf(backendDiagnostics == null ? List.of() : backendDiagnostics);
    }

    public PrepareTrace(
            boolean measured,
            long durationNs,
            int forwardStepCount,
            int backwardStepCount,
            BackendSelectionTrace backendSelection
    ) {
        this(measured, durationNs, forwardStepCount, backwardStepCount, backendSelection, List.of());
    }

    /**
     * Returns a trace marker for skipped or unavailable preparation tracing.
     *
     * @return skipped prepare trace
     */
    public static PrepareTrace skipped() {
        return new PrepareTrace(false, 0L, 0, 0, BackendSelectionTrace.empty(), List.of());
    }
}
