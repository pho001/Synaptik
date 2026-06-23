package graph.compile.planning.region.specialization;

import java.util.Objects;

/**
 * Structured metadata for a canonical scaled-dot-product-attention backward specialization candidate.
 *
 * <p>Node id fields that are not required by the selected output kind are {@code -1}.</p>
 */
public record SdpaBackwardSpecializationPayload(
        SdpaBackwardOutputKind outputKind,
        double scale,
        boolean hasMask,
        int weightsNodeId,
        int outGradNodeId,
        int queryNodeId,
        int keyNodeId,
        int valueNodeId,
        int maskNodeId
) implements RegionSpecializationPayload {
    public SdpaBackwardSpecializationPayload {
        outputKind = Objects.requireNonNull(outputKind, "outputKind cannot be null");
        if (!(scale > 0.0d)) {
            throw new IllegalArgumentException("scale must be positive: " + scale);
        }
        requireNodeId(weightsNodeId, "weightsNodeId");
        requireNodeId(outGradNodeId, "outGradNodeId");
        requireOptionalNodeId(queryNodeId, "queryNodeId");
        requireOptionalNodeId(keyNodeId, "keyNodeId");
        requireOptionalNodeId(valueNodeId, "valueNodeId");
        requireOptionalNodeId(maskNodeId, "maskNodeId");
        if (hasMask && maskNodeId < 0) {
            throw new IllegalArgumentException("masked SDPA backward payload requires maskNodeId");
        }
    }

    private static void requireNodeId(int nodeId, String name) {
        if (nodeId < 0) {
            throw new IllegalArgumentException(name + " must be >= 0");
        }
    }

    private static void requireOptionalNodeId(int nodeId, String name) {
        if (nodeId < -1) {
            throw new IllegalArgumentException(name + " must be >= -1");
        }
    }
}
