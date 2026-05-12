package graph.compile;

import graph.execution.trace.PartitionCompileTrace;
import graph.optimizer.partition.BackendCandidatePartition;
import graph.optimizer.partition.Partition;
import graph.optimizer.partition.PartitionPlan;

import java.util.List;

/**
 * Authoritative backend ownership planning artifact for one compile.
 */
public record BackendPlanningResult(
        List<BackendPlanningJob> jobs,
        List<Partition> partitions,
        List<PartitionPlan> backendPlans,
        List<BackendCandidatePartition> backendSelectionCandidates,
        PartitionCompileTrace trace,
        List<BackendPlanningDiagnostic> diagnostics
) {
    public BackendPlanningResult {
        jobs = List.copyOf(jobs == null ? List.of() : jobs);
        partitions = List.copyOf(partitions == null ? List.of() : partitions);
        backendPlans = List.copyOf(backendPlans == null ? List.of() : backendPlans);
        backendSelectionCandidates = List.copyOf(backendSelectionCandidates == null ? List.of() : backendSelectionCandidates);
        trace = trace == null ? PartitionCompileTrace.empty() : trace;
        diagnostics = List.copyOf(diagnostics == null ? List.of() : diagnostics);
    }

    public static BackendPlanningResult empty() {
        return new BackendPlanningResult(
                List.of(),
                List.of(),
                List.of(),
                List.of(),
                PartitionCompileTrace.empty(),
                List.of()
        );
    }
}
