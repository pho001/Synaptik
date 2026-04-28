package graph.execution.trace;

import backend.runtime.ExecutionMode;

import java.util.List;
import java.util.Objects;

/**
 * Run-stage diagnostics for one prepared execution.
 *
 * @param mode execution mode that ran
 * @param durationNs total run duration in nanoseconds
 * @param steps per-step trace metadata, empty when tracing was disabled
 */
public record RunTrace(
        ExecutionMode mode,
        long durationNs,
        List<ExecutionStepTrace> steps
) {
    public RunTrace {
        Objects.requireNonNull(mode, "mode cannot be null");
        steps = steps == null ? List.of() : List.copyOf(steps);
    }

    /**
     * Returns an empty run trace for a mode.
     *
     * @param mode execution mode
     * @return empty run trace
     */
    public static RunTrace empty(ExecutionMode mode) {
        return new RunTrace(mode, 0L, List.of());
    }
}
