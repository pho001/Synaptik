package backend.select;

import graph.execution.trace.BackendSelectionTrace;
import graph.optimizer.partition.PartitionPlan;

import java.util.List;

/**
 * Result of backend selection after partition planning.
 *
 * @param selectedPlans partition plans accepted for backend execution
 * @param trace diagnostic trace explaining backend selection decisions
 */
public record BackendSelectionResult(
        List<PartitionPlan> selectedPlans,
        BackendSelectionTrace trace
) {
    public BackendSelectionResult {
        selectedPlans = List.copyOf(selectedPlans == null ? List.of() : selectedPlans);
        trace = trace == null ? BackendSelectionTrace.empty() : trace;
    }

    /**
     * @return empty selection result with an empty trace
     */
    public static BackendSelectionResult empty() {
        return new BackendSelectionResult(List.of(), BackendSelectionTrace.empty());
    }
}
