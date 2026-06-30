package backend.metal.buffer;

import runtime.device.buffer.AcceleratorBufferLayout;
import runtime.device.buffer.AcceleratorBufferLayoutClass;
import runtime.device.buffer.AcceleratorBufferReasonCode;

import java.util.Arrays;
import java.util.Objects;

/**
 * Metal-specific layout policy for Java-side buffer preflight decisions.
 */
final class MetalLayoutPolicy {
    private MetalLayoutPolicy() {
    }

    enum Action {
        DIRECT_DENSE_BUFFER,
        DENSE_PHYSICAL_LOGICAL_VIEW,
        REJECT
    }

    record Decision(Action action, AcceleratorBufferReasonCode reasonCode, String reason) {
        Decision {
            Objects.requireNonNull(action, "action cannot be null");
            Objects.requireNonNull(reasonCode, "reasonCode cannot be null");
            reason = reason == null ? "" : reason;
        }

        boolean accepted() {
            return action != Action.REJECT;
        }

        boolean requiresDensePhysicalLogicalView() {
            return action == Action.DENSE_PHYSICAL_LOGICAL_VIEW;
        }
    }

    static Decision cpuUploadInput(AcceleratorBufferLayout layout) {
        Objects.requireNonNull(layout, "layout cannot be null");
        return switch (layout.layoutClass()) {
            case DENSE_CONTIGUOUS -> direct();
            case ZERO_OFFSET_VIEW, NON_ZERO_OFFSET_VIEW, PERMUTED_OR_STRIDED_VIEW ->
                    reject(
                            layout,
                            AcceleratorBufferReasonCode.INPUT_LAYOUT_UNSUPPORTED,
                            "cpuUploadRequires=DENSE_CONTIGUOUS"
                    );
            case BROADCAST_ZERO_STRIDE_VIEW -> reject(
                    layout,
                    AcceleratorBufferReasonCode.INPUT_LAYOUT_UNSUPPORTED,
                    "broadcast zero-stride layout is not supported by Metal buffer execution"
            );
            case UNSUPPORTED -> reject(
                    layout,
                    AcceleratorBufferReasonCode.INPUT_LAYOUT_UNSUPPORTED,
                    "unsupported layout class"
            );
        };
    }

    static Decision existingDeviceInput(AcceleratorBufferLayout layout) {
        Objects.requireNonNull(layout, "layout cannot be null");
        return switch (layout.layoutClass()) {
            case DENSE_CONTIGUOUS -> direct();
            case ZERO_OFFSET_VIEW, NON_ZERO_OFFSET_VIEW, PERMUTED_OR_STRIDED_VIEW -> densePhysicalLogicalView(layout);
            case BROADCAST_ZERO_STRIDE_VIEW -> reject(
                    layout,
                    AcceleratorBufferReasonCode.INPUT_LAYOUT_UNSUPPORTED,
                    "broadcast zero-stride layout is not supported by Metal buffer execution"
            );
            case UNSUPPORTED -> reject(
                    layout,
                    AcceleratorBufferReasonCode.INPUT_LAYOUT_UNSUPPORTED,
                    "unsupported layout class"
            );
        };
    }

    static Decision output(AcceleratorBufferLayout layout) {
        Objects.requireNonNull(layout, "layout cannot be null");
        return switch (layout.layoutClass()) {
            case DENSE_CONTIGUOUS -> direct();
            case ZERO_OFFSET_VIEW, NON_ZERO_OFFSET_VIEW, PERMUTED_OR_STRIDED_VIEW -> densePhysicalLogicalView(layout);
            case BROADCAST_ZERO_STRIDE_VIEW -> reject(
                    layout,
                    AcceleratorBufferReasonCode.OUTPUT_LAYOUT_UNSUPPORTED,
                    "broadcast zero-stride layout is not supported by Metal buffer execution"
            );
            case UNSUPPORTED -> reject(
                    layout,
                    AcceleratorBufferReasonCode.OUTPUT_LAYOUT_UNSUPPORTED,
                    "unsupported layout class"
            );
        };
    }

    private static Decision direct() {
        return new Decision(
                Action.DIRECT_DENSE_BUFFER,
                AcceleratorBufferReasonCode.BUFFER_BINDING_AVAILABLE,
                ""
        );
    }

    private static Decision densePhysicalLogicalView(AcceleratorBufferLayout layout) {
        return new Decision(
                Action.DENSE_PHYSICAL_LOGICAL_VIEW,
                AcceleratorBufferReasonCode.BUFFER_BINDING_AVAILABLE,
                reason(Action.DENSE_PHYSICAL_LOGICAL_VIEW, layout, "dense physical logical-view layout")
        );
    }

    private static Decision reject(
            AcceleratorBufferLayout layout,
            AcceleratorBufferReasonCode reasonCode,
            String detail
    ) {
        return new Decision(Action.REJECT, reasonCode, reason(Action.REJECT, layout, detail));
    }

    private static String reason(Action action, AcceleratorBufferLayout layout, String detail) {
        return "policyAction=" + action
                + ", layoutClass=" + layout.layoutClass()
                + ", shape=" + Arrays.toString(layout.shape())
                + ", storageOffset=" + layout.storageOffset()
                + ", strides=" + Arrays.toString(layout.strides())
                + ", " + detail;
    }
}
