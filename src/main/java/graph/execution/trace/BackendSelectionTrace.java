package graph.execution.trace;

import java.util.List;

public record BackendSelectionTrace(
        int totalCandidates,
        int selectedCount,
        int rejectedCount,
        List<BackendSelectionDecisionTrace> decisions
) {
    public BackendSelectionTrace {
        totalCandidates = Math.max(0, totalCandidates);
        selectedCount = Math.max(0, selectedCount);
        rejectedCount = Math.max(0, rejectedCount);
        decisions = List.copyOf(decisions == null ? List.of() : decisions);
    }

    public static BackendSelectionTrace empty() {
        return new BackendSelectionTrace(0, 0, 0, List.of());
    }
}
