package graph.compile;

import graph.execution.trace.PartitionCompileTrace;
import graph.optimizer.partition.PlannedPartition;

import java.util.List;

/**
 * Authoritative backend ownership planning artifact for one compile.
 */
public record BackendPlanningResult(
        List<BackendPlanningJob> jobs,
        List<PlannedPartition> plannedPartitions,
        PartitionCompileTrace trace,
        List<BackendPlanningDiagnostic> diagnostics
) {
    public BackendPlanningResult {
        jobs = List.copyOf(jobs == null ? List.of() : jobs);
        plannedPartitions = List.copyOf(plannedPartitions == null ? List.of() : plannedPartitions);
        trace = trace == null ? PartitionCompileTrace.empty() : trace;
        diagnostics = List.copyOf(diagnostics == null ? List.of() : diagnostics);
    }

    public static BackendPlanningResult empty() {
        return new BackendPlanningResult(
                List.of(),
                List.of(),
                PartitionCompileTrace.empty(),
                List.of()
        );
    }
}
