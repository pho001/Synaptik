package graph.execution.trace;

import java.util.List;

/**
 * Summary of backend selection during preparation.
 *
 * @param totalCandidates number of candidate backend partitions considered
 * @param selectedCount number of selected candidates
 * @param rejectedCount number of rejected candidates
 * @param decisions detailed selection decisions
 */
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

    /**
     * Returns an empty backend-selection trace.
     *
     * @return empty trace
     */
    public static BackendSelectionTrace empty() {
        return new BackendSelectionTrace(0, 0, 0, List.of());
    }
}
