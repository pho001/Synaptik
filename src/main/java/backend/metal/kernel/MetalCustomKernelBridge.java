package backend.metal.kernel;

import backend.metal.lowering.MetalPartitionPlan;

/**
 * Optional SPI for custom Metal kernels inside selected Metal GPU regions.
 *
 * <p>This seam is backend-internal. It consumes lowered Metal region metadata and deliberately
 * does not expose or require public Tensor device-residency APIs.</p>
 */
public interface MetalCustomKernelBridge {
    /**
     * Returns custom-kernel route capabilities without throwing.
     */
    MetalCustomKernelCapabilities capabilities();

    /**
     * Prepares a custom-kernel executable for a lowered Metal plan.
     */
    default MetalCustomKernelExecutable compile(MetalPartitionPlan plan) {
        return MetalCustomKernelExecutable.unavailable(capabilities().reason());
    }

    /**
     * Returns the canonical unavailable custom-kernel bridge.
     */
    static MetalCustomKernelBridge unavailable() {
        return UnavailableMetalCustomKernelBridge.INSTANCE;
    }
}
