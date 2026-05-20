package graph.execution.trace;

import backend.ComputeBackend;
import backend.accelerator.lowering.GpuLoweredRegionManifest;
import graph.compile.planning.partition.cost.AcceleratorPartitionScoreModel;

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
 * @param costSummary static materialization-aware cost summary, if available
 * @param finalists bounded rejected finalist summaries
 * @param gpuLoweredRegionManifest selected lowered GPU region manifest, if this is an accepted GPU decision
 */
public record BackendSelectionDecisionTrace(
        int anchorNodeId,
        List<Integer> nodeIds,
        List<ComputeBackend> compatibleBackends,
        boolean selected,
        ComputeBackend selectedBackend,
        String reason,
        long estimatedWork,
        AcceleratorPartitionScoreModel.MaterializationCostSummary costSummary,
        List<PartitionDecisionTrace.CandidateCostTrace> finalists,
        GpuLoweredRegionManifest gpuLoweredRegionManifest
) {
    public BackendSelectionDecisionTrace(
            int anchorNodeId,
            List<Integer> nodeIds,
            List<ComputeBackend> compatibleBackends,
            boolean selected,
            ComputeBackend selectedBackend,
            String reason,
            long estimatedWork,
            AcceleratorPartitionScoreModel.MaterializationCostSummary costSummary,
            List<PartitionDecisionTrace.CandidateCostTrace> finalists
    ) {
        this(
                anchorNodeId,
                nodeIds,
                compatibleBackends,
                selected,
                selectedBackend,
                reason,
                estimatedWork,
                costSummary,
                finalists,
                null
        );
    }

    public BackendSelectionDecisionTrace(
            int anchorNodeId,
            List<Integer> nodeIds,
            List<ComputeBackend> compatibleBackends,
            boolean selected,
            ComputeBackend selectedBackend,
            String reason,
            long estimatedWork
    ) {
        this(
                anchorNodeId,
                nodeIds,
                compatibleBackends,
                selected,
                selectedBackend,
                reason,
                estimatedWork,
                null,
                List.of(),
                null
        );
    }

    public BackendSelectionDecisionTrace {
        nodeIds = List.copyOf(nodeIds == null ? List.of() : nodeIds);
        compatibleBackends = List.copyOf(compatibleBackends == null ? List.of() : compatibleBackends);
        reason = reason == null ? "" : reason;
        estimatedWork = Math.max(0L, estimatedWork);
        finalists = List.copyOf(finalists == null ? List.of() : finalists).stream()
                .limit(3)
                .toList();
        if (!selected) {
            gpuLoweredRegionManifest = null;
        }
    }
}
