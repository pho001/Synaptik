package graph.optimizer.state;

import java.util.List;

/**
 * Lightweight optimizer diagnostics carried through {@link OptimizerState}.
 *
 * @param events ordered textual events recorded by optimizer stages
 */
public record OptimizerTrace(
        List<String> events
) {
    public OptimizerTrace {
        events = List.copyOf(events == null ? List.of() : events);
    }

    /**
     * Returns an empty trace.
     *
     * @return trace with no events
     */
    public static OptimizerTrace empty() {
        return new OptimizerTrace(List.of());
    }
}
