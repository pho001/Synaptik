package backend.accelerator.lowering;

import backend.ComputeBackend;

import java.util.List;
import java.util.Map;

/**
 * Java-side manifest for a selected GPU region lowered to backend primitives.
 *
 * <p>This is trace and planning metadata. It deliberately wraps the native bridge DAG
 * instead of changing the Metal/CUDA native ABI.</p>
 *
 * @param regionId stable region id for trace/report references
 * @param backend selected backend that owns the region
 * @param anchorNodeId compiled node that anchors execution
 * @param orderedNodeIds original graph nodes selected for the region
 * @param externalInputNodeIds graph node ids read from outside the region
 * @param outputNodeIds graph node ids produced by the region
 * @param selectedRegionLength number of original graph nodes retained in the selected region
 * @param originalOps original operation metadata and primitive mapping
 * @param loweredPrimitives lowered primitive metadata
 * @param inputAssumptions value assumptions for region inputs
 * @param outputAssumptions value assumptions for region outputs
 * @param fusedSummary region-internal fused/compound summary placeholder
 * @param rejections rejection, fallback, materialization, and shortening metadata
 * @param candidateSpan candidate-shortening metadata
 * @param backendExtensions backend-specific string metadata outside the shared core
 */
public record GpuLoweredRegionManifest(
        String regionId,
        ComputeBackend backend,
        int anchorNodeId,
        List<Integer> orderedNodeIds,
        List<Integer> externalInputNodeIds,
        List<Integer> outputNodeIds,
        int selectedRegionLength,
        List<GpuLoweredRegionOriginalOp> originalOps,
        List<GpuLoweredPrimitiveManifest> loweredPrimitives,
        List<GpuLoweredRegionValueAssumption> inputAssumptions,
        List<GpuLoweredRegionValueAssumption> outputAssumptions,
        GpuCompoundRegionSummary fusedSummary,
        List<GpuLoweredRegionRejection> rejections,
        GpuLoweredRegionCandidateSpan candidateSpan,
        Map<String, String> backendExtensions
) {
    public GpuLoweredRegionManifest {
        regionId = regionId == null ? "" : regionId;
        backend = backend == null ? ComputeBackend.CPU : backend;
        orderedNodeIds = List.copyOf(orderedNodeIds == null ? List.of() : orderedNodeIds);
        externalInputNodeIds = List.copyOf(externalInputNodeIds == null ? List.of() : externalInputNodeIds);
        outputNodeIds = List.copyOf(outputNodeIds == null ? List.of() : outputNodeIds);
        selectedRegionLength = Math.max(0, selectedRegionLength);
        originalOps = List.copyOf(originalOps == null ? List.of() : originalOps);
        loweredPrimitives = List.copyOf(loweredPrimitives == null ? List.of() : loweredPrimitives);
        inputAssumptions = List.copyOf(inputAssumptions == null ? List.of() : inputAssumptions);
        outputAssumptions = List.copyOf(outputAssumptions == null ? List.of() : outputAssumptions);
        fusedSummary = fusedSummary == null ? GpuCompoundRegionSummary.none(backend, orderedNodeIds) : fusedSummary;
        rejections = List.copyOf(rejections == null ? List.of() : rejections);
        candidateSpan = candidateSpan == null ? GpuLoweredRegionCandidateSpan.none(orderedNodeIds) : candidateSpan;
        backendExtensions = Map.copyOf(backendExtensions == null ? Map.of() : backendExtensions);
    }
}
