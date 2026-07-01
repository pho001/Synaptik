package backend.accelerator.lowering;

/**
 * Rejection or fallback evidence attributed to an original op, primitive, fused subpattern, or boundary.
 *
 * @param level attribution level such as original_op, primitive, fused_subpattern, or partition_boundary
 * @param originalNodeId original node id, or -1 when not applicable
 * @param primitiveId lowered primitive id, if applicable
 * @param fusedPatternType fused/compound pattern type, if applicable
 * @param reason stable reason code
 * @param detail compact human-readable detail
 */
public record GpuLoweredPartitionRejection(
        String level,
        int originalNodeId,
        String primitiveId,
        String fusedPatternType,
        GpuLoweringUnsupportedReason reason,
        String detail
) {
    public GpuLoweredPartitionRejection {
        level = level == null ? "UNKNOWN" : level;
        primitiveId = primitiveId == null ? "" : primitiveId;
        fusedPatternType = fusedPatternType == null ? "" : fusedPatternType;
        reason = reason == null ? GpuLoweringUnsupportedReason.DAG_PRIMITIVE_UNSUPPORTED : reason;
        detail = detail == null ? "" : detail;
    }
}
