package backend.metal.exec;

import java.util.List;
import java.util.Objects;

/**
 * Prepare-time route evidence for one Metal executable.
 *
 * <p>This model is Metal-internal routing metadata. It does not select graph partitions
 * or change public tensor residency semantics.</p>
 */
public record MetalRouteDecision(
        MetalExecutionRoute selectedRoute,
        List<MetalExecutionRoute> rejectedRoutes,
        MetalRouteReasonCode reasonCode,
        String detail,
        long estimatedWork,
        long estimatedRouteCost,
        long estimatedCopyCost,
        boolean bridgeAvailable,
        boolean executableAvailable,
        boolean bufferAbiSupported,
        boolean customKernelAvailable,
        boolean nativeCopyCostKnown
) {
    public MetalRouteDecision {
        selectedRoute = Objects.requireNonNullElse(selectedRoute, MetalExecutionRoute.CPU_FALLBACK);
        rejectedRoutes = List.copyOf(rejectedRoutes == null ? List.of() : rejectedRoutes);
        reasonCode = Objects.requireNonNullElse(reasonCode, MetalRouteReasonCode.CPU_FALLBACK);
        detail = detail == null ? "" : detail;
        estimatedWork = Math.max(0L, estimatedWork);
        estimatedRouteCost = Math.max(0L, estimatedRouteCost);
        estimatedCopyCost = Math.max(-1L, estimatedCopyCost);
    }

    public static MetalRouteDecision unavailable(
            MetalExecutionRoute selectedRoute,
            MetalRouteReasonCode reasonCode,
            String detail,
            long estimatedWork,
            boolean required
    ) {
        return new MetalRouteDecision(
                required ? MetalExecutionRoute.UNAVAILABLE_REQUIRED : selectedRoute,
                List.of(MetalExecutionRoute.MPS_GRAPH, MetalExecutionRoute.CUSTOM_KERNEL),
                required ? MetalRouteReasonCode.UNAVAILABLE_REQUIRED : reasonCode,
                detail,
                estimatedWork,
                estimatedWork,
                -1L,
                false,
                false,
                false,
                false,
                false
        );
    }
}
