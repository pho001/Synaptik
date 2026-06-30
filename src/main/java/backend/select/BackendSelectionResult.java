package backend.select;

import trace.prepare.BackendSelectionTrace;
import planning.partition.PartitionPlan;
import planning.partition.PlannedPartition;

import java.util.List;

/**
 * Result of backend selection after partition planning.
 *
 * @param selectedPartitions planned partitions accepted for backend execution
 * @param trace diagnostic trace explaining backend selection decisions
 */
public record BackendSelectionResult(
        List<PlannedPartition> selectedPartitions,
        BackendSelectionTrace trace
) {
    public BackendSelectionResult {
        selectedPartitions = List.copyOf(selectedPartitions == null ? List.of() : selectedPartitions);
        trace = trace == null ? BackendSelectionTrace.empty() : trace;
    }

    /**
     * Returns selected backend plans derived from selected planned partitions.
     *
     * @return selected plans
     */
    public List<PartitionPlan> selectedPlans() {
        return selectedPartitions.stream()
                .map(PlannedPartition::plan)
                .filter(java.util.Objects::nonNull)
                .toList();
    }

    /**
     * @return empty selection result with an empty trace
     */
    public static BackendSelectionResult empty() {
        return new BackendSelectionResult(List.of(), BackendSelectionTrace.empty());
    }
}
