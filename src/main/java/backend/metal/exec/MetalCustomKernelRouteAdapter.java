package backend.metal.exec;

import backend.accelerator.lowering.GpuLoweredPrimitiveManifest;
import backend.metal.kernel.MetalCustomKernelCapabilities;
import backend.metal.kernel.MetalCustomKernelExecutable;
import backend.metal.lowering.MetalPartitionPlan;

import java.util.List;

/**
 * Converts optional custom-kernel capability state into route evidence.
 */
final class MetalCustomKernelRouteAdapter {
    private MetalCustomKernelRouteAdapter() {
    }

    static CustomKernelEvidence evaluate(
            MetalPartitionPlan plan,
            MetalCustomKernelCapabilities capabilities,
            MetalCustomKernelExecutable executable
    ) {
        MetalCustomKernelCapabilities caps = capabilities == null
                ? MetalCustomKernelCapabilities.unavailable("custom Metal kernel bridge unavailable")
                : capabilities;
        List<String> primitiveIds = loweredPrimitiveIds(plan);
        if (!caps.available()) {
            return new CustomKernelEvidence(
                    false,
                    false,
                    caps.reasonCode(),
                    caps.reason(),
                    primitiveIds
            );
        }
        MetalCustomKernelExecutable compiled = executable == null
                ? MetalCustomKernelExecutable.unavailable("custom Metal kernel executable unavailable")
                : executable;
        if (!compiled.available()) {
            return new CustomKernelEvidence(
                    false,
                    true,
                    compiled.reasonCode(),
                    compiled.reason(),
                    primitiveIds
            );
        }
        return new CustomKernelEvidence(
                true,
                true,
                MetalRouteReasonCode.MPS_GRAPH_SELECTED,
                "custom Metal kernel executable available but route selection is capability-gated",
                primitiveIds
        );
    }

    private static List<String> loweredPrimitiveIds(MetalPartitionPlan plan) {
        if (plan == null || plan.manifest() == null) {
            return List.of();
        }
        return plan.manifest().loweredPrimitives().stream()
                .map(GpuLoweredPrimitiveManifest::primitiveId)
                .filter(id -> id != null && !id.isBlank())
                .toList();
    }

    record CustomKernelEvidence(
            boolean available,
            boolean candidate,
            MetalRouteReasonCode reasonCode,
            String reason,
            List<String> loweredPrimitiveIds
    ) {
        CustomKernelEvidence {
            reasonCode = reasonCode == null ? MetalRouteReasonCode.CUSTOM_KERNEL_UNAVAILABLE : reasonCode;
            reason = reason == null ? "" : reason;
            loweredPrimitiveIds = List.copyOf(loweredPrimitiveIds == null ? List.of() : loweredPrimitiveIds);
        }
    }
}
