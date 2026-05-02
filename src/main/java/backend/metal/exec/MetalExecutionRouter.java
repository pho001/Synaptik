package backend.metal.exec;

import backend.accelerator.buffer.AcceleratorBufferReasonCode;
import backend.accelerator.lowering.GpuLoweredRegionManifest;
import backend.metal.bridge.MetalMpsBridgeCapabilities;
import backend.metal.lowering.MetalPartitionPlan;
import config.runtime.AcceleratorBackendConfig;
import config.runtime.AcceleratorBufferBindingMode;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Prepare-time router for execution strategies inside selected Metal regions.
 */
public final class MetalExecutionRouter {
    private MetalExecutionRouter() {
    }

    public static MetalRouteDecision decide(
            MetalPartitionPlan plan,
            MetalMpsBridgeCapabilities capabilities,
            AcceleratorBackendConfig backendConfig,
            TransportEvidence transport
    ) {
        Objects.requireNonNull(plan, "plan cannot be null");
        TransportEvidence evidence = transport == null ? TransportEvidence.unavailable(plan.estimatedWork()) : transport;
        AcceleratorBackendConfig config = backendConfig == null ? AcceleratorBackendConfig.defaults() : backendConfig;
        MetalMpsBridgeCapabilities caps = capabilities == null
                ? MetalMpsBridgeCapabilities.unavailable(null, "capabilities unavailable")
                : capabilities;
        long work = Math.max(0L, plan.estimatedWork());
        List<MetalExecutionRoute> rejected = new ArrayList<>();
        rejected.add(MetalExecutionRoute.CUSTOM_KERNEL);

        if (!config.enabled()) {
            return decision(
                    evidence.required() ? MetalExecutionRoute.UNAVAILABLE_REQUIRED : MetalExecutionRoute.CPU_FALLBACK,
                    rejected,
                    evidence.required() ? MetalRouteReasonCode.UNAVAILABLE_REQUIRED : MetalRouteReasonCode.CPU_FALLBACK,
                    "Metal backend config disabled",
                    work,
                    evidence,
                    caps
            );
        }
        if (!evidence.bridgeAvailable() || !evidence.contextAvailable() || !evidence.executableAvailable()) {
            rejected.add(MetalExecutionRoute.MPS_GRAPH);
            return decision(
                    evidence.required() ? MetalExecutionRoute.UNAVAILABLE_REQUIRED : MetalExecutionRoute.CPU_FALLBACK,
                    rejected,
                    evidence.required() ? MetalRouteReasonCode.UNAVAILABLE_REQUIRED : MetalRouteReasonCode.BRIDGE_UNAVAILABLE,
                    evidence.reason(),
                    work,
                    evidence,
                    caps
            );
        }
        if (!evidence.staticDTypeLegal()) {
            rejected.add(MetalExecutionRoute.MPS_GRAPH);
            return decision(
                    evidence.required() ? MetalExecutionRoute.UNAVAILABLE_REQUIRED : MetalExecutionRoute.CPU_FALLBACK,
                    rejected,
                    evidence.required() ? MetalRouteReasonCode.UNAVAILABLE_REQUIRED : MetalRouteReasonCode.UNSUPPORTED_DTYPE,
                    evidence.reason(),
                    work,
                    evidence,
                    caps
            );
        }

        return switch (evidence.preferredPath()) {
            case BUFFER_BINDING -> decision(
                    MetalExecutionRoute.MPS_GRAPH,
                    rejected,
                    MetalRouteReasonCode.MPS_GRAPH_SELECTED,
                    mpsGraphDetail(plan.manifest(), evidence),
                    work,
                    evidence,
                    caps
            );
            case TENSOR_ARRAY -> decision(
                    MetalExecutionRoute.TENSOR_ARRAY,
                    rejected,
                    tensorArrayReason(evidence.reasonCode()),
                    evidence.reason(),
                    work,
                    evidence,
                    caps
            );
            case STATIC_CPU_FALLBACK -> {
                rejected.add(MetalExecutionRoute.MPS_GRAPH);
                yield decision(
                        MetalExecutionRoute.CPU_FALLBACK,
                        rejected,
                        fallbackReason(evidence.reasonCode()),
                        evidence.reason(),
                        work,
                        evidence,
                        caps
                );
            }
            case UNAVAILABLE_REQUIRED -> {
                rejected.add(MetalExecutionRoute.MPS_GRAPH);
                yield decision(
                        MetalExecutionRoute.UNAVAILABLE_REQUIRED,
                        rejected,
                        MetalRouteReasonCode.UNAVAILABLE_REQUIRED,
                        evidence.reason(),
                        work,
                        evidence,
                        caps
                );
            }
        };
    }

    private static MetalRouteDecision decision(
            MetalExecutionRoute route,
            List<MetalExecutionRoute> rejectedRoutes,
            MetalRouteReasonCode reasonCode,
            String detail,
            long estimatedWork,
            TransportEvidence evidence,
            MetalMpsBridgeCapabilities capabilities
    ) {
        boolean bufferAbiSupported = evidence.bufferAbiSupported() || capabilities.bufferExecutionSupported();
        return new MetalRouteDecision(
                route,
                rejectedRoutes,
                reasonCode,
                detail,
                estimatedWork,
                estimatedRouteCost(estimatedWork, route, evidence),
                -1L,
                evidence.bridgeAvailable(),
                evidence.executableAvailable(),
                bufferAbiSupported,
                false,
                false
        );
    }

    private static long estimatedRouteCost(long estimatedWork, MetalExecutionRoute route, TransportEvidence evidence) {
        long copyPenalty = switch (route) {
            case MPS_GRAPH -> evidence.preferredPath() == TransportPath.BUFFER_BINDING ? 0L : estimatedWork;
            case TENSOR_ARRAY -> estimatedWork * 2L;
            case CPU_FALLBACK, UNAVAILABLE_REQUIRED -> estimatedWork;
            case CUSTOM_KERNEL -> 0L;
        };
        return saturatingAdd(estimatedWork, copyPenalty);
    }

    private static long saturatingAdd(long left, long right) {
        long result = left + right;
        if (((left ^ result) & (right ^ result)) < 0) {
            return Long.MAX_VALUE;
        }
        return Math.max(0L, result);
    }

    private static String mpsGraphDetail(GpuLoweredRegionManifest manifest, TransportEvidence evidence) {
        String regionId = manifest == null ? "" : manifest.regionId();
        String base = "MPSGraph selected via " + evidence.preferredPath()
                + "; custom kernel route unavailable"
                + "; native copy cost unknown";
        return regionId == null || regionId.isBlank()
                ? base
                : base + "; regionId=" + regionId;
    }

    private static MetalRouteReasonCode tensorArrayReason(AcceleratorBufferReasonCode reasonCode) {
        if (reasonCode == AcceleratorBufferReasonCode.NATIVE_BUFFER_ABI_UNAVAILABLE) {
            return MetalRouteReasonCode.BUFFER_ABI_UNAVAILABLE;
        }
        if (reasonCode == AcceleratorBufferReasonCode.BELOW_MINIMUM_WORK) {
            return MetalRouteReasonCode.INSUFFICIENT_WORK;
        }
        return MetalRouteReasonCode.TENSOR_ARRAY_FALLBACK;
    }

    private static MetalRouteReasonCode fallbackReason(AcceleratorBufferReasonCode reasonCode) {
        if (reasonCode == AcceleratorBufferReasonCode.INPUT_DTYPE_UNSUPPORTED) {
            return MetalRouteReasonCode.UNSUPPORTED_DTYPE;
        }
        if (reasonCode == AcceleratorBufferReasonCode.INPUT_LAYOUT_UNSUPPORTED
                || reasonCode == AcceleratorBufferReasonCode.OUTPUT_LAYOUT_UNSUPPORTED) {
            return MetalRouteReasonCode.UNSUPPORTED_LAYOUT;
        }
        if (reasonCode == AcceleratorBufferReasonCode.BELOW_MINIMUM_WORK) {
            return MetalRouteReasonCode.INSUFFICIENT_WORK;
        }
        if (reasonCode == AcceleratorBufferReasonCode.BRIDGE_UNAVAILABLE) {
            return MetalRouteReasonCode.BRIDGE_UNAVAILABLE;
        }
        return MetalRouteReasonCode.CPU_FALLBACK;
    }

    public enum TransportPath {
        BUFFER_BINDING,
        TENSOR_ARRAY,
        STATIC_CPU_FALLBACK,
        UNAVAILABLE_REQUIRED
    }

    public record TransportEvidence(
            TransportPath preferredPath,
            AcceleratorBufferBindingMode mode,
            AcceleratorBufferReasonCode reasonCode,
            String reason,
            boolean bridgeAvailable,
            boolean contextAvailable,
            boolean executableAvailable,
            boolean bufferAbiSupported,
            boolean staticDTypeLegal,
            boolean containsForwardAttentionDag,
            long estimatedWork,
            long minimumEstimatedWork
    ) {
        public TransportEvidence {
            preferredPath = preferredPath == null ? TransportPath.STATIC_CPU_FALLBACK : preferredPath;
            mode = mode == null ? AcceleratorBufferBindingMode.AUTO : mode;
            reasonCode = reasonCode == null ? AcceleratorBufferReasonCode.NOT_EVALUATED : reasonCode;
            reason = reason == null ? "" : reason;
            estimatedWork = Math.max(0L, estimatedWork);
            minimumEstimatedWork = Math.max(0L, minimumEstimatedWork);
        }

        public static TransportEvidence unavailable(long estimatedWork) {
            return new TransportEvidence(
                    TransportPath.STATIC_CPU_FALLBACK,
                    AcceleratorBufferBindingMode.AUTO,
                    AcceleratorBufferReasonCode.BRIDGE_UNAVAILABLE,
                    "transport evidence unavailable",
                    false,
                    false,
                    false,
                    false,
                    false,
                    false,
                    estimatedWork,
                    0L
            );
        }

        public boolean required() {
            return mode == AcceleratorBufferBindingMode.REQUIRE;
        }
    }
}
