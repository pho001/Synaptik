package planning.partition.specialization;

import planning.partition.execution.ExecutionUnit;

import java.util.List;

/**
 * Result of partition specialization candidate planning.
 *
 * @param accepted whether specialized units should replace structural planning
 * @param units accepted specialized units
 * @param traceEvents trace/debug events emitted while planning candidates
 */
public record PartitionSpecializationResult(
        boolean accepted,
        List<ExecutionUnit> units,
        List<String> traceEvents
) {
    public PartitionSpecializationResult {
        units = List.copyOf(units == null ? List.of() : units);
        traceEvents = List.copyOf(traceEvents == null ? List.of() : traceEvents);
    }

    /**
     * Empty specialization result.
     *
     * @return result with no accepted units
     */
    public static PartitionSpecializationResult empty() {
        return new PartitionSpecializationResult(false, List.of(), List.of());
    }
}
