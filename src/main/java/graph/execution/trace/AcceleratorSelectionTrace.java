package graph.execution.trace;

import java.util.List;

public record AcceleratorSelectionTrace(
        int totalCandidates,
        int selectedCount,
        int rejectedCount,
        List<AcceleratorSelectionDecisionTrace> decisions
) {
    public AcceleratorSelectionTrace {
        totalCandidates = Math.max(0, totalCandidates);
        selectedCount = Math.max(0, selectedCount);
        rejectedCount = Math.max(0, rejectedCount);
        decisions = List.copyOf(decisions == null ? List.of() : decisions);
    }

    public static AcceleratorSelectionTrace empty() {
        return new AcceleratorSelectionTrace(0, 0, 0, List.of());
    }
}
