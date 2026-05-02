package backend.metal.exec;

import backend.metal.kernel.MetalCustomKernelCapabilities;
import backend.metal.kernel.MetalCustomKernelCandidate;
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
        MetalCustomKernelCandidate candidate = MetalCustomKernelCandidate.evaluate(plan);
        if (!caps.available()) {
            return new CustomKernelEvidence(
                    false,
                    false,
                    caps.reasonCode(),
                    caps.reason(),
                    candidate.primitiveIds(),
                    candidate.kernelId()
            );
        }
        if (!candidate.supported()) {
            return new CustomKernelEvidence(
                    false,
                    false,
                    candidate.reasonCode(),
                    candidate.reason(),
                    candidate.primitiveIds(),
                    candidate.kernelId()
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
                    candidate.primitiveIds(),
                    candidate.kernelId()
            );
        }
        return new CustomKernelEvidence(
                true,
                true,
                MetalRouteReasonCode.CUSTOM_KERNEL_SELECTED,
                "custom Metal kernel executable available",
                candidate.primitiveIds(),
                candidate.kernelId()
        );
    }

    record CustomKernelEvidence(
            boolean available,
            boolean candidate,
            MetalRouteReasonCode reasonCode,
            String reason,
            List<String> loweredPrimitiveIds,
            String kernelId
    ) {
        CustomKernelEvidence {
            reasonCode = reasonCode == null ? MetalRouteReasonCode.CUSTOM_KERNEL_UNAVAILABLE : reasonCode;
            reason = reason == null ? "" : reason;
            loweredPrimitiveIds = List.copyOf(loweredPrimitiveIds == null ? List.of() : loweredPrimitiveIds);
            kernelId = kernelId == null ? "" : kernelId;
        }
    }
}
