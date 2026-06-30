package planning.region.specialization;

import planning.region.ExecutionUnit;

import java.util.List;

/**
 * Result of region specialization candidate planning.
 *
 * @param accepted whether specialized units should replace structural planning
 * @param units accepted specialized units
 * @param traceEvents trace/debug events emitted while planning candidates
 */
public record RegionSpecializationResult(
        boolean accepted,
        List<ExecutionUnit> units,
        List<String> traceEvents
) {
    public RegionSpecializationResult {
        units = List.copyOf(units == null ? List.of() : units);
        traceEvents = List.copyOf(traceEvents == null ? List.of() : traceEvents);
    }

    /**
     * Empty specialization result.
     *
     * @return result with no accepted units
     */
    public static RegionSpecializationResult empty() {
        return new RegionSpecializationResult(false, List.of(), List.of());
    }
}
