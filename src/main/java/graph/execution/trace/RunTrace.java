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
 * @param cpuMaterializations CPU materialization requests observed during execution
 */
public record RunTrace(
        ExecutionMode mode,
        long durationNs,
        List<ExecutionStepTrace> steps,
        List<CpuMaterializationTrace> cpuMaterializations
) {
    public RunTrace {
        Objects.requireNonNull(mode, "mode cannot be null");
        steps = steps == null ? List.of() : List.copyOf(steps);
        cpuMaterializations = cpuMaterializations == null ? List.of() : List.copyOf(cpuMaterializations);
    }

    /**
     * Creates run diagnostics without CPU materialization entries.
     *
     * @param mode execution mode that ran
     * @param durationNs total run duration in nanoseconds
     * @param steps per-step trace metadata
     */
    public RunTrace(ExecutionMode mode, long durationNs, List<ExecutionStepTrace> steps) {
        this(mode, durationNs, steps, List.of());
    }

    /**
     * Returns an empty run trace for a mode.
     *
     * @param mode execution mode
     * @return empty run trace
     */
    public static RunTrace empty(ExecutionMode mode) {
        return new RunTrace(mode, 0L, List.of(), List.of());
    }
}
