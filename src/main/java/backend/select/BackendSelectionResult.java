package backend.select;

import graph.execution.trace.BackendSelectionTrace;
import graph.optimizer.partition.PartitionPlan;

import java.util.List;

public record BackendSelectionResult(
        List<PartitionPlan> selectedPlans,
        BackendSelectionTrace trace
) {
    public BackendSelectionResult {
        selectedPlans = List.copyOf(selectedPlans == null ? List.of() : selectedPlans);
        trace = trace == null ? BackendSelectionTrace.empty() : trace;
    }

    public static BackendSelectionResult empty() {
        return new BackendSelectionResult(List.of(), BackendSelectionTrace.empty());
    }
}
