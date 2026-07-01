package backend.accelerator.lowering;

import backend.contract.ComputeBackend;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Java-side manifest for a selected GPU partition lowered to backend primitives.
 *
 * <p>This is trace and planning metadata. It deliberately wraps the native bridge DAG
 * instead of changing the Metal/CUDA native ABI.</p>
 *
 * @param partitionId stable partition id for trace/report references
 * @param backend selected backend that owns the partition
 * @param anchorNodeId compiled node that anchors execution
 * @param orderedNodeIds original graph nodes selected for the partition
 * @param externalInputNodeIds graph node ids read from outside the partition
 * @param outputNodeIds graph node ids produced by the partition
 * @param selectedPartitionLength number of original graph nodes retained in the selected partition
 * @param originalOps original operation metadata and primitive mapping
 * @param loweredPrimitives lowered primitive metadata
 * @param inputAssumptions value assumptions for partition inputs
 * @param outputAssumptions value assumptions for partition outputs
 * @param fusedSummary partition-internal fused/compound summary placeholder
 * @param rejections rejection, fallback, materialization, and shortening metadata
 * @param candidateSpan candidate-shortening metadata
 * @param backendExtensions backend-specific string metadata outside the shared core
 */
public record GpuLoweredPartitionManifest(
        String partitionId,
        ComputeBackend backend,
        int anchorNodeId,
        List<Integer> orderedNodeIds,
        List<Integer> externalInputNodeIds,
        List<Integer> outputNodeIds,
        int selectedPartitionLength,
        List<GpuLoweredPartitionOriginalOp> originalOps,
        List<GpuLoweredPrimitiveManifest> loweredPrimitives,
        List<GpuLoweredPartitionValueAssumption> inputAssumptions,
        List<GpuLoweredPartitionValueAssumption> outputAssumptions,
        GpuCompoundPartitionSummary fusedSummary,
        List<GpuFusionSubpatternSummary> fusedSubpatterns,
        List<GpuLoweredPartitionRejection> rejections,
        GpuLoweredPartitionCandidateSpan candidateSpan,
        Map<String, String> backendExtensions
) {
    public GpuLoweredPartitionManifest {
        partitionId = partitionId == null ? "" : partitionId;
        backend = backend == null ? ComputeBackend.CPU : backend;
        orderedNodeIds = List.copyOf(orderedNodeIds == null ? List.of() : orderedNodeIds);
        externalInputNodeIds = List.copyOf(externalInputNodeIds == null ? List.of() : externalInputNodeIds);
        outputNodeIds = List.copyOf(outputNodeIds == null ? List.of() : outputNodeIds);
        selectedPartitionLength = Math.max(0, selectedPartitionLength);
        originalOps = List.copyOf(originalOps == null ? List.of() : originalOps);
        loweredPrimitives = List.copyOf(loweredPrimitives == null ? List.of() : loweredPrimitives);
        inputAssumptions = List.copyOf(inputAssumptions == null ? List.of() : inputAssumptions);
        outputAssumptions = List.copyOf(outputAssumptions == null ? List.of() : outputAssumptions);
        fusedSummary = fusedSummary == null ? GpuCompoundPartitionSummary.none(backend, orderedNodeIds) : fusedSummary;
        fusedSubpatterns = normalizeFusedSubpatterns(fusedSubpatterns, fusedSummary, loweredPrimitives);
        rejections = List.copyOf(rejections == null ? List.of() : rejections);
        candidateSpan = candidateSpan == null ? GpuLoweredPartitionCandidateSpan.none(orderedNodeIds) : candidateSpan;
        backendExtensions = backendExtensions == null
                ? Map.of()
                : Collections.unmodifiableMap(new LinkedHashMap<>(backendExtensions));
    }

    public GpuLoweredPartitionManifest(
            String partitionId,
            ComputeBackend backend,
            int anchorNodeId,
            List<Integer> orderedNodeIds,
            List<Integer> externalInputNodeIds,
            List<Integer> outputNodeIds,
            int selectedPartitionLength,
            List<GpuLoweredPartitionOriginalOp> originalOps,
            List<GpuLoweredPrimitiveManifest> loweredPrimitives,
            List<GpuLoweredPartitionValueAssumption> inputAssumptions,
            List<GpuLoweredPartitionValueAssumption> outputAssumptions,
            GpuCompoundPartitionSummary fusedSummary,
            List<GpuLoweredPartitionRejection> rejections,
            GpuLoweredPartitionCandidateSpan candidateSpan,
            Map<String, String> backendExtensions
    ) {
        this(
                partitionId,
                backend,
                anchorNodeId,
                orderedNodeIds,
                externalInputNodeIds,
                outputNodeIds,
                selectedPartitionLength,
                originalOps,
                loweredPrimitives,
                inputAssumptions,
                outputAssumptions,
                fusedSummary,
                null,
                rejections,
                candidateSpan,
                backendExtensions
        );
    }

    private static List<GpuFusionSubpatternSummary> normalizeFusedSubpatterns(
            List<GpuFusionSubpatternSummary> fusedSubpatterns,
            GpuCompoundPartitionSummary fusedSummary,
            List<GpuLoweredPrimitiveManifest> loweredPrimitives
    ) {
        if (fusedSubpatterns != null && !fusedSubpatterns.isEmpty()) {
            return List.copyOf(fusedSubpatterns);
        }
        if (fusedSummary == null || fusedSummary.patternType() == GpuCompoundPatternType.NONE) {
            return List.of();
        }
        List<String> primitiveIds = loweredPrimitives == null
                ? List.of()
                : loweredPrimitives.stream()
                        .map(GpuLoweredPrimitiveManifest::primitiveId)
                        .toList();
        if (fusedSummary.supported()) {
            return List.of(GpuFusionSubpatternSummary.supported(
                    fusedSummary.patternType(),
                    fusedSummary.orderedNodeIds(),
                    primitiveIds,
                    fusedSummary.detail()
            ));
        }
        return List.of(GpuFusionSubpatternSummary.unsupported(
                fusedSummary.patternType(),
                fusedSummary.reason(),
                fusedSummary.orderedNodeIds(),
                fusedSummary.detail()
        ));
    }
}
