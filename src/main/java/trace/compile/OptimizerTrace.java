package trace.compile;

import java.util.List;

/**
 * Lightweight optimizer diagnostics carried through {@link OptimizerState}.
 *
 * @param events ordered textual events recorded by optimizer stages
 * @param costExplanations ordered structured cost explanations recorded by optimizer stages
 */
public record OptimizerTrace(
        List<String> events,
        List<CostExplanationTrace> costExplanations
) {
    public OptimizerTrace {
        events = List.copyOf(events == null ? List.of() : events);
        costExplanations = List.copyOf(costExplanations == null ? List.of() : costExplanations);
    }

    public OptimizerTrace(List<String> events) {
        this(events, List.of());
    }

    /**
     * Returns an empty trace.
     *
     * @return trace with no events
     */
    public static OptimizerTrace empty() {
        return new OptimizerTrace(List.of(), List.of());
    }

    /**
     * Returns a copy with one appended textual event.
     *
     * @param event event to append
     * @return updated trace
     */
    public OptimizerTrace withEvent(String event) {
        if (event == null || event.isBlank()) {
            return this;
        }
        java.util.ArrayList<String> next = new java.util.ArrayList<>(events);
        next.add(event);
        return new OptimizerTrace(next, costExplanations);
    }

    /**
     * Returns a copy with one appended cost explanation.
     *
     * @param explanation explanation to append
     * @return updated trace
     */
    public OptimizerTrace withCostExplanation(CostExplanationTrace explanation) {
        if (explanation == null) {
            return this;
        }
        java.util.ArrayList<CostExplanationTrace> next = new java.util.ArrayList<>(costExplanations);
        next.add(explanation);
        return new OptimizerTrace(events, next);
    }
}
