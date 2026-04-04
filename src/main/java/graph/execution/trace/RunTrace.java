package graph.execution.trace;

import backend.runtime.ExecutionMode;

import java.util.List;
import java.util.Objects;

public record RunTrace(
        ExecutionMode mode,
        long durationNs,
        List<ExecutionStepTrace> steps
) {
    public RunTrace {
        Objects.requireNonNull(mode, "mode cannot be null");
        steps = steps == null ? List.of() : List.copyOf(steps);
    }

    public static RunTrace empty(ExecutionMode mode) {
        return new RunTrace(mode, 0L, List.of());
    }
}
