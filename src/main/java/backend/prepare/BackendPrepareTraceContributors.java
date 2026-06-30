package backend.prepare;

import backend.contract.ComputeBackend;
import backend.lowering.LoweringInput;
import backend.select.BackendSelectionResult;
import trace.prepare.BackendPrepareDiagnosticTrace;
import graph.compile.planning.partition.PlannedPartition;

import java.util.LinkedHashMap;
import java.util.List;

final class BackendPrepareTraceContributors {
    private BackendPrepareTraceContributors() {
    }

    static List<BackendPrepareDiagnosticTrace> diagnostics(
            BackendSelectionResult selection,
            LoweringInput loweringInput
    ) {
        return List.of(
                selectionDiagnostics(selection),
                loweringDiagnostics(selection, loweringInput)
        );
    }

    private static BackendPrepareDiagnosticTrace selectionDiagnostics(BackendSelectionResult selection) {
        LinkedHashMap<String, Object> attrs = new LinkedHashMap<>();
        List<PlannedPartition> selected = selection == null ? List.of() : selection.selectedPartitions();
        attrs.put("selectedPartitionCount", selected.size());
        attrs.put("selectedBackends", selected.stream()
                .map(PlannedPartition::plan)
                .filter(java.util.Objects::nonNull)
                .map(plan -> plan.backend().name())
                .distinct()
                .toList());
        attrs.put("acceleratorPartitionCount", selected.stream()
                .map(PlannedPartition::plan)
                .filter(java.util.Objects::nonNull)
                .map(plan -> plan.backend())
                .filter(backend -> backend != ComputeBackend.CPU)
                .count());
        return new BackendPrepareDiagnosticTrace("backend-selection", attrs);
    }

    private static BackendPrepareDiagnosticTrace loweringDiagnostics(
            BackendSelectionResult selection,
            LoweringInput loweringInput
    ) {
        LinkedHashMap<String, Object> attrs = new LinkedHashMap<>();
        attrs.put("loweringInputPresent", loweringInput != null);
        attrs.put("optimizedRegionCount", loweringInput == null ? 0 : loweringInput.optimizedRegions().size());
        attrs.put("selectedPlanCount", selection == null ? 0 : selection.selectedPlans().size());
        attrs.put("memoryPlanPresent", loweringInput != null && loweringInput.memoryPlan() != null);
        return new BackendPrepareDiagnosticTrace("backend-lowering", attrs);
    }
}
