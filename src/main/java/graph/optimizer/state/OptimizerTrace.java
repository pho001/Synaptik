package graph.optimizer.state;

import java.util.List;

public record OptimizerTrace(
        List<String> events
) {
    public OptimizerTrace {
        events = List.copyOf(events == null ? List.of() : events);
    }

    public static OptimizerTrace empty() {
        return new OptimizerTrace(List.of());
    }
}
