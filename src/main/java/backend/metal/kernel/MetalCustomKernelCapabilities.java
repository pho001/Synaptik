package backend.metal.kernel;

import backend.metal.exec.MetalRouteReasonCode;

import java.util.Objects;

/**
 * Capability state for the optional custom Metal kernel route.
 */
public record MetalCustomKernelCapabilities(
        boolean available,
        MetalRouteReasonCode reasonCode,
        String reason
) {
    public MetalCustomKernelCapabilities {
        reasonCode = Objects.requireNonNullElse(reasonCode, MetalRouteReasonCode.CUSTOM_KERNEL_UNAVAILABLE);
        reason = reason == null ? "" : reason;
    }

    public static MetalCustomKernelCapabilities unavailable(String reason) {
        return new MetalCustomKernelCapabilities(
                false,
                MetalRouteReasonCode.CUSTOM_KERNEL_UNAVAILABLE,
                reason == null || reason.isBlank()
                        ? "custom Metal kernel bridge unavailable"
                        : reason
        );
    }
}
