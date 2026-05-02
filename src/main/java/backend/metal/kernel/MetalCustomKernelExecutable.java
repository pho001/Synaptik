package backend.metal.kernel;

import backend.metal.exec.MetalRouteReasonCode;

import java.util.List;
import java.util.Objects;

/**
 * Prepared custom-kernel executable descriptor.
 *
 * <p>The descriptor is intentionally independent of public tensor device APIs. Concrete native handle
 * ownership will stay inside the Metal backend when a real custom kernel route is implemented.</p>
 */
public record MetalCustomKernelExecutable(
        boolean available,
        String kernelId,
        List<String> primitiveIds,
        MetalRouteReasonCode reasonCode,
        String reason
) {
    public MetalCustomKernelExecutable {
        kernelId = kernelId == null ? "" : kernelId;
        primitiveIds = List.copyOf(primitiveIds == null ? List.of() : primitiveIds);
        reasonCode = Objects.requireNonNullElse(reasonCode, MetalRouteReasonCode.CUSTOM_KERNEL_UNAVAILABLE);
        reason = reason == null ? "" : reason;
    }

    public static MetalCustomKernelExecutable unavailable(String reason) {
        return new MetalCustomKernelExecutable(
                false,
                "",
                List.of(),
                MetalRouteReasonCode.CUSTOM_KERNEL_UNAVAILABLE,
                reason == null || reason.isBlank()
                        ? "custom Metal kernel executable unavailable"
                        : reason
        );
    }
}
