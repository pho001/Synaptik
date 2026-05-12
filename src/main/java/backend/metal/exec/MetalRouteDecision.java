package backend.metal.exec;

import graph.optimizer.cost.CostComponent;
import graph.optimizer.cost.CostScore;

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
        List<MetalRouteReasonCode> rejectedReasonCodes,
        List<String> rejectedRouteReasons,
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
        rejectedReasonCodes = List.copyOf(rejectedReasonCodes == null ? List.of() : rejectedReasonCodes);
        rejectedRouteReasons = List.copyOf(rejectedRouteReasons == null ? List.of() : rejectedRouteReasons);
        reasonCode = Objects.requireNonNullElse(reasonCode, MetalRouteReasonCode.CPU_FALLBACK);
        detail = detail == null ? "" : detail;
        estimatedWork = Math.max(0L, estimatedWork);
        estimatedRouteCost = Math.max(0L, estimatedRouteCost);
        estimatedCopyCost = Math.max(-1L, estimatedCopyCost);
    }

    /**
     * Exports this Metal route decision through the shared cost vocabulary.
     *
     * <p>This is report-only. It does not rerun route selection, touch native state,
     * allocate buffers, or change Metal execution behavior.</p>
     *
     * @return shared cost score explanation input
     */
    public CostScore toCostScore() {
        return CostScore.of(
                "MetalBackendRouteCostModel",
                "metal-prepared-execution-route",
                List.of(
                        CostComponent.lowerIsBetter(
                                "estimatedRouteCost",
                                estimatedRouteCost,
                                "estimated route cost selected by MetalExecutionRouter"
                        ),
                        CostComponent.informational(
                                "estimatedWork",
                                estimatedWork,
                                "backend work estimate for the selected Metal region"
                        ),
                        CostComponent.lowerIsBetter(
                                "estimatedCopyCost",
                                estimatedCopyCost < 0L ? 0.0d : estimatedCopyCost,
                                estimatedCopyCost < 0L ? "native copy cost is unknown" : "estimated native copy/materialization cost"
                        ),
                        CostComponent.lowerIsBetter(
                                "tensorArrayFallback",
                                selectedRoute == MetalExecutionRoute.TENSOR_ARRAY ? 1.0d : 0.0d,
                                "selected route uses legacy tensor-array bridge fallback"
                        ),
                        CostComponent.lowerIsBetter(
                                "cpuFallback",
                                selectedRoute == MetalExecutionRoute.CPU_FALLBACK ? 1.0d : 0.0d,
                                "selected route exits the Metal region to CPU fallback"
                        ),
                        CostComponent.lowerIsBetter(
                                "unavailableRequired",
                                selectedRoute == MetalExecutionRoute.UNAVAILABLE_REQUIRED ? 1.0d : 0.0d,
                                "required Metal execution was unavailable"
                        ),
                        CostComponent.lowerIsBetter(
                                "rejectedRouteCount",
                                rejectedRoutes.size(),
                                "number of Metal route alternatives rejected at prepare time"
                        ),
                        CostComponent.informational(
                                "selectedRoute",
                                0.0d,
                                selectedRoute.name()
                        ),
                        CostComponent.informational(
                                "bufferAbiSupported",
                                bufferAbiSupported ? 1.0d : 0.0d,
                                "native buffer ABI support observed at prepare time"
                        ),
                        CostComponent.informational(
                                "bridgeAvailable",
                                bridgeAvailable ? 1.0d : 0.0d,
                                "Metal bridge availability observed at prepare time"
                        ),
                        CostComponent.informational(
                                "executableAvailable",
                                executableAvailable ? 1.0d : 0.0d,
                                "compiled Metal executable availability observed at prepare time"
                        ),
                        CostComponent.informational(
                                "customKernelAvailable",
                                customKernelAvailable ? 1.0d : 0.0d,
                                "scoped custom Metal kernel route availability"
                        ),
                        CostComponent.informational(
                                "nativeCopyCostKnown",
                                nativeCopyCostKnown ? 1.0d : 0.0d,
                                "whether native copy cost was known when routing"
                        )
                )
        );
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
                List.of(reasonCode, MetalRouteReasonCode.CUSTOM_KERNEL_UNAVAILABLE),
                List.of(detail, "custom Metal kernel bridge unavailable"),
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
