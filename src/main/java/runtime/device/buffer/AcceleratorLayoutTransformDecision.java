package runtime.device.buffer;

import operations.Operation;

import java.util.Objects;

/**
 * Result of classifying a GPU layout transform request.
 */
public record AcceleratorLayoutTransformDecision(
        String backendId,
        int sourceNodeId,
        int targetNodeId,
        Operation.OpType opType,
        AcceleratorLayoutTransformKind kind,
        boolean accepted,
        AcceleratorBufferReasonCode reasonCode,
        String reason,
        AcceleratorBufferLayout sourceLayout,
        AcceleratorBufferLayout targetLayout
) {
    public AcceleratorLayoutTransformDecision {
        backendId = requireNonBlank(backendId, "backendId");
        Objects.requireNonNull(opType, "opType cannot be null");
        Objects.requireNonNull(kind, "kind cannot be null");
        Objects.requireNonNull(reasonCode, "reasonCode cannot be null");
        reason = requireNonBlank(reason, "reason");
        Objects.requireNonNull(sourceLayout, "sourceLayout cannot be null");
        Objects.requireNonNull(targetLayout, "targetLayout cannot be null");
    }

    public static AcceleratorLayoutTransformDecision metadataOnlyView(
            AcceleratorLayoutTransformRequest request,
            String reason
    ) {
        return accepted(request,
                AcceleratorLayoutTransformKind.METADATA_ONLY_VIEW,
                AcceleratorBufferReasonCode.GPU_LAYOUT_VIEW_BINDING_AVAILABLE,
                reason);
    }

    public static AcceleratorLayoutTransformDecision denseGpuMaterialization(
            AcceleratorLayoutTransformRequest request,
            String reason
    ) {
        return accepted(request,
                AcceleratorLayoutTransformKind.DENSE_GPU_MATERIALIZATION,
                AcceleratorBufferReasonCode.GPU_LAYOUT_DENSE_MATERIALIZATION_AVAILABLE,
                reason);
    }

    public static AcceleratorLayoutTransformDecision broadcastGpuMaterialization(
            AcceleratorLayoutTransformRequest request,
            String reason
    ) {
        return accepted(request,
                AcceleratorLayoutTransformKind.BROADCAST_GPU_MATERIALIZATION,
                AcceleratorBufferReasonCode.GPU_LAYOUT_BROADCAST_MATERIALIZATION_AVAILABLE,
                reason);
    }

    public static AcceleratorLayoutTransformDecision stridedNativeCompute(
            AcceleratorLayoutTransformRequest request,
            String reason
    ) {
        return accepted(request,
                AcceleratorLayoutTransformKind.STRIDED_NATIVE_COMPUTE,
                AcceleratorBufferReasonCode.GPU_LAYOUT_STRIDED_NATIVE_COMPUTE_AVAILABLE,
                reason);
    }

    public static AcceleratorLayoutTransformDecision rejected(
            AcceleratorLayoutTransformRequest request,
            AcceleratorBufferReasonCode reasonCode,
            String reason
    ) {
        return new AcceleratorLayoutTransformDecision(
                request.backendId(),
                request.sourceNodeId(),
                request.targetNodeId(),
                request.opType(),
                AcceleratorLayoutTransformKind.UNSUPPORTED,
                false,
                reasonCode,
                reason,
                request.sourceLayout(),
                request.targetLayout()
        );
    }

    private static AcceleratorLayoutTransformDecision accepted(
            AcceleratorLayoutTransformRequest request,
            AcceleratorLayoutTransformKind kind,
            AcceleratorBufferReasonCode reasonCode,
            String reason
    ) {
        return new AcceleratorLayoutTransformDecision(
                request.backendId(),
                request.sourceNodeId(),
                request.targetNodeId(),
                request.opType(),
                kind,
                true,
                reasonCode,
                reason,
                request.sourceLayout(),
                request.targetLayout()
        );
    }

    private static String requireNonBlank(String value, String label) {
        Objects.requireNonNull(value, label + " cannot be null");
        if (value.isBlank()) {
            throw new IllegalArgumentException(label + " cannot be blank");
        }
        return value;
    }
}
