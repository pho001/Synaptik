package backend.metal.kernel;

import backend.metal.lowering.MetalPartitionPlan;
import backend.metal.buffer.MetalBufferBinding;
import backend.metal.bridge.MetalMpsBridgeContext;
import backend.metal.bridge.MetalMpsBridgeExecutionStats;

import java.util.List;

/**
 * Optional SPI for custom Metal kernels inside selected Metal GPU partitions.
 *
 * <p>This seam is backend-internal. It consumes lowered Metal partition metadata and deliberately
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
     * Executes a compiled custom kernel against explicit native buffer bindings.
     */
    default MetalMpsBridgeExecutionStats executeBuffers(
            MetalMpsBridgeContext context,
            MetalCustomKernelExecutable executable,
            List<MetalBufferBinding> externalInputs,
            List<MetalBufferBinding> outputs
    ) {
        throw new UnsupportedOperationException("custom Metal kernel bridge does not support native buffer execution.");
    }

    /**
     * Returns the canonical unavailable custom-kernel bridge.
     */
    static MetalCustomKernelBridge unavailable() {
        return UnavailableMetalCustomKernelBridge.INSTANCE;
    }
}
