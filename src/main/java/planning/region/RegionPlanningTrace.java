package planning.region;

import java.util.List;

/**
 * Diagnostics emitted by region planning.
 *
 * @param events ordered textual events
 */
public record RegionPlanningTrace(
        List<String> events
) {
    public RegionPlanningTrace {
        events = List.copyOf(events == null ? List.of() : events);
    }

    /**
     * Returns an empty region planning trace.
     *
     * @return empty trace
     */
    public static RegionPlanningTrace empty() {
        return new RegionPlanningTrace(List.of());
    }
}
