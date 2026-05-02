package backend.accelerator.buffer;

import backend.memory.DeviceBufferBinding;
import operations.Operation;

import java.util.Objects;

/**
 * Pure classifier for deciding whether a layout operation can stay on an accelerator path.
 */
public final class AcceleratorLayoutTransformPlanner {
    private AcceleratorLayoutTransformPlanner() {
    }

    public static AcceleratorLayoutTransformDecision decide(AcceleratorLayoutTransformRequest request) {
        Objects.requireNonNull(request, "request cannot be null");

        DeviceBufferBinding sourceBinding = request.sourceBinding();
        if (sourceBinding == null || !sourceBinding.available()) {
            return AcceleratorLayoutTransformDecision.rejected(
                    request,
                    AcceleratorBufferReasonCode.GPU_LAYOUT_SOURCE_BINDING_UNAVAILABLE,
                    "source binding unavailable for GPU layout transform");
        }
        if (!request.backendId().equals(sourceBinding.backendId())) {
            return AcceleratorLayoutTransformDecision.rejected(
                    request,
                    AcceleratorBufferReasonCode.GPU_LAYOUT_BACKEND_MISMATCH,
                    "backend mismatch for GPU layout transform: request=" + request.backendId()
                            + ", source=" + sourceBinding.backendId());
        }
        if (unsupportedLayout(request.sourceLayout()) || unsupportedLayout(request.targetLayout())) {
            return AcceleratorLayoutTransformDecision.rejected(
                    request,
                    AcceleratorBufferReasonCode.NATIVE_LAYOUT_METADATA_UNSUPPORTED,
                    "native layout metadata unsupported for GPU layout transform");
        }

        Operation.OpType opType = request.opType();
        if (isMetadataOnlyView(opType)) {
            return AcceleratorLayoutTransformDecision.metadataOnlyView(
                    request,
                    "metadata-only view binding available for " + opType);
        }
        if (opType == Operation.OpType.RESHAPE) {
            if (request.sourceLayout().layoutClass() == AcceleratorBufferLayoutClass.DENSE_CONTIGUOUS) {
                return AcceleratorLayoutTransformDecision.metadataOnlyView(
                        request,
                        "metadata-only view binding available for contiguous-source reshape");
            }
            if (isBroadcastToDenseRepair(request)) {
                return AcceleratorLayoutTransformDecision.broadcastGpuMaterialization(
                        request,
                        "broadcast GPU materialization available for non-contiguous reshape");
            }
            if (!isDenseTarget(request)) {
                return AcceleratorLayoutTransformDecision.rejected(
                        request,
                        AcceleratorBufferReasonCode.GPU_LAYOUT_TRANSFORM_UNSUPPORTED,
                        "GPU layout transform unsupported for non-dense reshape target");
            }
            return AcceleratorLayoutTransformDecision.denseGpuMaterialization(
                    request,
                    "dense GPU materialization available for non-contiguous reshape");
        }
        if (opType == Operation.OpType.CONTIGUOUS) {
            if (isBroadcastToDenseRepair(request)) {
                return AcceleratorLayoutTransformDecision.broadcastGpuMaterialization(
                        request,
                        "broadcast GPU materialization available for contiguous layout transform");
            }
            if (request.sourceLayout().layoutClass() == AcceleratorBufferLayoutClass.BROADCAST_ZERO_STRIDE_VIEW) {
                return AcceleratorLayoutTransformDecision.rejected(
                        request,
                        AcceleratorBufferReasonCode.GPU_LAYOUT_BROADCAST_MATERIALIZATION_UNSUPPORTED,
                        "broadcast GPU materialization requires dense contiguous target layout");
            }
            if (!isDenseTarget(request)) {
                return AcceleratorLayoutTransformDecision.rejected(
                        request,
                        AcceleratorBufferReasonCode.GPU_LAYOUT_TRANSFORM_UNSUPPORTED,
                        "GPU layout transform unsupported for non-dense contiguous target");
            }
            return AcceleratorLayoutTransformDecision.denseGpuMaterialization(
                    request,
                    "dense GPU materialization available for contiguous layout transform");
        }
        if (requiresStridedNativeCompute(request)) {
            return AcceleratorLayoutTransformDecision.rejected(
                    request,
                    AcceleratorBufferReasonCode.GPU_LAYOUT_STRIDED_NATIVE_COMPUTE_UNSUPPORTED,
                    "direct strided native compute unsupported for " + opType);
        }
        return AcceleratorLayoutTransformDecision.rejected(
                request,
                AcceleratorBufferReasonCode.GPU_LAYOUT_TRANSFORM_UNSUPPORTED,
                "GPU layout transform unsupported for " + opType);
    }

    private static boolean isMetadataOnlyView(Operation.OpType opType) {
        return switch (opType) {
            case NOOP, SELECT, PERMUTE, EXPAND, EXPAND_DIMS, SQUEEZE -> true;
            default -> false;
        };
    }

    private static boolean isBroadcastToDenseRepair(AcceleratorLayoutTransformRequest request) {
        return request.sourceLayout().layoutClass() == AcceleratorBufferLayoutClass.BROADCAST_ZERO_STRIDE_VIEW
                && isDenseTarget(request);
    }

    private static boolean isDenseTarget(AcceleratorLayoutTransformRequest request) {
        return request.targetLayout().layoutClass() == AcceleratorBufferLayoutClass.DENSE_CONTIGUOUS;
    }

    private static boolean requiresStridedNativeCompute(AcceleratorLayoutTransformRequest request) {
        return request.sourceLayout().layoutClass() == AcceleratorBufferLayoutClass.PERMUTED_OR_STRIDED_VIEW
                || request.sourceLayout().layoutClass() == AcceleratorBufferLayoutClass.ZERO_OFFSET_VIEW
                || request.sourceLayout().layoutClass() == AcceleratorBufferLayoutClass.NON_ZERO_OFFSET_VIEW;
    }

    private static boolean unsupportedLayout(AcceleratorBufferLayout layout) {
        return layout.layoutClass() == AcceleratorBufferLayoutClass.UNSUPPORTED;
    }
}
