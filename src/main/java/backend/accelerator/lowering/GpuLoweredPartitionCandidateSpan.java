package backend.accelerator.lowering;

import java.util.List;

/**
 * Candidate-span evidence when a GPU partition is shortened before execution.
 *
 * @param originalCandidateNodeIds node ids considered by the original candidate
 * @param acceptedNodeIds node ids retained in the selected partition
 * @param rejectedOriginalNodeId first original node that forced shortening, or -1
 * @param rejectedPrimitiveId lowered primitive that forced shortening, if known
 * @param reason stable shortening reason
 */
public record GpuLoweredPartitionCandidateSpan(
        List<Integer> originalCandidateNodeIds,
        List<Integer> acceptedNodeIds,
        int rejectedOriginalNodeId,
        String rejectedPrimitiveId,
        GpuLoweringUnsupportedReason reason
) {
    public GpuLoweredPartitionCandidateSpan {
        originalCandidateNodeIds = List.copyOf(originalCandidateNodeIds == null ? List.of() : originalCandidateNodeIds);
        acceptedNodeIds = List.copyOf(acceptedNodeIds == null ? List.of() : acceptedNodeIds);
        rejectedPrimitiveId = rejectedPrimitiveId == null ? "" : rejectedPrimitiveId;
        reason = reason == null ? GpuLoweringUnsupportedReason.SUPPORTED : reason;
    }

    /**
     * Returns an empty candidate span for a partition that was not shortened.
     */
    public static GpuLoweredPartitionCandidateSpan none(List<Integer> acceptedNodeIds) {
        List<Integer> accepted = List.copyOf(acceptedNodeIds == null ? List.of() : acceptedNodeIds);
        return new GpuLoweredPartitionCandidateSpan(
                accepted,
                accepted,
                -1,
                "",
                GpuLoweringUnsupportedReason.SUPPORTED
        );
    }
}
