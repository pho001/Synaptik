package backend.accelerator.lowering;

import backend.ComputeBackend;
import backend.accelerator.dag.AcceleratorDagNode;
import backend.accelerator.dag.AcceleratorDagSpec;

import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Result of lowering a candidate partition to accelerator bridge inputs.
 *
 * @param computeNodeId compiled-node id that anchors the lowered partition
 * @param matMulSpec optional legacy matmul descriptor retained for bridge compatibility
 * @param dagSpec backend-neutral lowered DAG consumed by native graph bridges
 * @param estimatedWork planner cost estimate for backend selection
 * @param compoundSummary fused/compound region summary used for trace metadata
 * @param manifest Java-side lowered-region manifest used for trace/report metadata
 */
public record AcceleratorSubgraphLoweringResult(
        int computeNodeId,
        AcceleratorMatMulSpec matMulSpec,
        AcceleratorDagSpec dagSpec,
        long estimatedWork,
        GpuCompoundRegionSummary compoundSummary,
        GpuLoweredRegionManifest manifest
) {
    public AcceleratorSubgraphLoweringResult {
        Objects.requireNonNull(dagSpec, "dagSpec cannot be null");
        estimatedWork = Math.max(0L, estimatedWork);
        compoundSummary = compoundSummary == null
                ? GpuCompoundRegionSummary.none(null, List.of())
                : compoundSummary;
        manifest = manifest == null
                ? defaultManifest(computeNodeId, dagSpec, compoundSummary)
                : manifest;
    }

    public AcceleratorSubgraphLoweringResult(
            int computeNodeId,
            AcceleratorMatMulSpec matMulSpec,
            AcceleratorDagSpec dagSpec,
            long estimatedWork,
            GpuCompoundRegionSummary compoundSummary
    ) {
        this(computeNodeId, matMulSpec, dagSpec, estimatedWork, compoundSummary, null);
    }

    public AcceleratorSubgraphLoweringResult(
            int computeNodeId,
            AcceleratorMatMulSpec matMulSpec,
            AcceleratorDagSpec dagSpec,
            long estimatedWork
    ) {
        this(computeNodeId, matMulSpec, dagSpec, estimatedWork, null);
    }

    private static GpuLoweredRegionManifest defaultManifest(
            int computeNodeId,
            AcceleratorDagSpec dagSpec,
            GpuCompoundRegionSummary compoundSummary
    ) {
        ComputeBackend backend = compoundSummary == null ? ComputeBackend.CPU : compoundSummary.backend();
        List<Integer> orderedNodeIds = dagSpec == null
                ? List.of()
                : dagSpec.nodes().stream().map(AcceleratorDagNode::nodeId).toList();
        List<Integer> outputNodeIds = dagSpec == null ? List.of() : dagSpec.outputNodeIds();
        return new GpuLoweredRegionManifest(
                "gpu-region-" + computeNodeId,
                backend,
                computeNodeId,
                orderedNodeIds,
                List.of(),
                outputNodeIds,
                orderedNodeIds.size(),
                List.of(),
                List.of(),
                List.of(),
                List.of(),
                compoundSummary,
                List.of(),
                List.of(),
                GpuLoweredRegionCandidateSpan.none(orderedNodeIds),
                Map.of()
        );
    }
}
