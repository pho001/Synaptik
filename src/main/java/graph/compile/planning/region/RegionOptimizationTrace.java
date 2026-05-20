package graph.compile.planning.region;

import java.util.List;

/**
 * Diagnostics emitted by region optimization.
 *
 * @param events ordered textual events
 */
public record RegionOptimizationTrace(
        List<String> events
) {
    public RegionOptimizationTrace {
        events = List.copyOf(events == null ? List.of() : events);
    }

    /**
     * Returns an empty region optimization trace.
     *
     * @return empty trace
     */
    public static RegionOptimizationTrace empty() {
        return new RegionOptimizationTrace(List.of());
    }
}
