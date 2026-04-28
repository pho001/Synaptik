package graph.execution.trace;

import backend.ComputeBackend;

import java.util.List;

/**
 * Decision for one backend selection candidate.
 *
 * @param anchorNodeId candidate anchor node id
 * @param nodeIds node ids covered by the candidate
 * @param compatibleBackends backends compatible with the candidate
 * @param selected whether the candidate was selected
 * @param selectedBackend backend chosen for execution, if selected
 * @param reason diagnostic reason
 * @param estimatedWork backend work estimate
 */
public record BackendSelectionDecisionTrace(
        int anchorNodeId,
        List<Integer> nodeIds,
        List<ComputeBackend> compatibleBackends,
        boolean selected,
        ComputeBackend selectedBackend,
        String reason,
        long estimatedWork
) {
    public BackendSelectionDecisionTrace {
        nodeIds = List.copyOf(nodeIds == null ? List.of() : nodeIds);
        compatibleBackends = List.copyOf(compatibleBackends == null ? List.of() : compatibleBackends);
        reason = reason == null ? "" : reason;
        estimatedWork = Math.max(0L, estimatedWork);
    }
}
