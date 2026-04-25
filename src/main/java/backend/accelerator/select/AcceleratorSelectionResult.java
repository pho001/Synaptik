package backend.accelerator.select;

import graph.execution.trace.AcceleratorSelectionTrace;
import graph.optimizer.partition.AcceleratorPartitionPlan;

import java.util.List;

public record AcceleratorSelectionResult(
        List<AcceleratorPartitionPlan> selectedPlans,
        AcceleratorSelectionTrace trace
) {
    public AcceleratorSelectionResult {
        selectedPlans = List.copyOf(selectedPlans == null ? List.of() : selectedPlans);
        trace = trace == null ? AcceleratorSelectionTrace.empty() : trace;
    }

    public static AcceleratorSelectionResult empty() {
        return new AcceleratorSelectionResult(List.of(), AcceleratorSelectionTrace.empty());
    }
}
