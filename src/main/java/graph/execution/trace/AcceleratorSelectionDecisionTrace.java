package graph.execution.trace;

import backend.ComputeBackend;

import java.util.List;

public record AcceleratorSelectionDecisionTrace(
        int anchorNodeId,
        List<Integer> nodeIds,
        List<ComputeBackend> compatibleBackends,
        boolean selected,
        ComputeBackend selectedBackend,
        String reason,
        long estimatedWork
) {
    public AcceleratorSelectionDecisionTrace {
        nodeIds = List.copyOf(nodeIds == null ? List.of() : nodeIds);
        compatibleBackends = List.copyOf(compatibleBackends == null ? List.of() : compatibleBackends);
        reason = reason == null ? "" : reason;
        estimatedWork = Math.max(0L, estimatedWork);
    }
}
