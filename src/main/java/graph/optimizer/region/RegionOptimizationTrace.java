package graph.optimizer.region;

import java.util.List;

public record RegionOptimizationTrace(
        List<String> events
) {
    public RegionOptimizationTrace {
        events = List.copyOf(events == null ? List.of() : events);
    }

    public static RegionOptimizationTrace empty() {
        return new RegionOptimizationTrace(List.of());
    }
}
