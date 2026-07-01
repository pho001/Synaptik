package backend.accelerator.lowering;

import java.util.List;
import java.util.Objects;

/**
 * Partition-internal GPU fusion subpattern metadata.
 *
 * <p>This is trace and lowering metadata. It is not a public operation model and
 * does not represent CPU {@code Operation.OpType.FUSED}.</p>
 */
public record GpuFusionSubpatternSummary(
        GpuCompoundPatternType patternType,
        boolean supported,
        List<Integer> originalOperationNodeIds,
        List<String> loweredPrimitiveIds,
        int loweredPrimitiveCount,
        GpuLoweringUnsupportedReason reason,
        String detail
) {
    public GpuFusionSubpatternSummary {
        patternType = patternType == null ? GpuCompoundPatternType.NONE : patternType;
        originalOperationNodeIds = List.copyOf(originalOperationNodeIds == null ? List.of() : originalOperationNodeIds);
        loweredPrimitiveIds = List.copyOf(loweredPrimitiveIds == null ? List.of() : loweredPrimitiveIds);
        loweredPrimitiveCount = loweredPrimitiveCount < 0 ? loweredPrimitiveIds.size() : loweredPrimitiveCount;
        reason = reason == null ? GpuLoweringUnsupportedReason.COMPOUND_PATTERN_UNSUPPORTED : reason;
        detail = detail == null ? "" : detail;
        if (supported && reason != GpuLoweringUnsupportedReason.SUPPORTED) {
            throw new IllegalArgumentException("supported GPU fusion subpattern must use SUPPORTED reason");
        }
        if (!supported && reason == GpuLoweringUnsupportedReason.SUPPORTED) {
            throw new IllegalArgumentException("unsupported GPU fusion subpattern must not use SUPPORTED reason");
        }
    }

    /**
     * Creates supported GPU fusion subpattern metadata.
     */
    public static GpuFusionSubpatternSummary supported(
            GpuCompoundPatternType patternType,
            List<Integer> originalOperationNodeIds,
            List<String> loweredPrimitiveIds,
            String detail
    ) {
        Objects.requireNonNull(patternType, "patternType cannot be null");
        return new GpuFusionSubpatternSummary(
                patternType,
                true,
                originalOperationNodeIds,
                loweredPrimitiveIds,
                loweredPrimitiveIds == null ? 0 : loweredPrimitiveIds.size(),
                GpuLoweringUnsupportedReason.SUPPORTED,
                detail
        );
    }

    /**
     * Creates unsupported GPU fusion subpattern metadata.
     */
    public static GpuFusionSubpatternSummary unsupported(
            GpuCompoundPatternType patternType,
            GpuLoweringUnsupportedReason reason,
            List<Integer> originalOperationNodeIds,
            String detail
    ) {
        Objects.requireNonNull(patternType, "patternType cannot be null");
        return new GpuFusionSubpatternSummary(
                patternType,
                false,
                originalOperationNodeIds,
                List.of(),
                0,
                reason,
                detail
        );
    }
}
