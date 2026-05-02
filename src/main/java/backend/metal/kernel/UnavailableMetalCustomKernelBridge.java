package backend.metal.kernel;

import backend.metal.lowering.MetalPartitionPlan;

/**
 * Stable unavailable implementation for the custom Metal kernel route.
 */
final class UnavailableMetalCustomKernelBridge implements MetalCustomKernelBridge {
    static final UnavailableMetalCustomKernelBridge INSTANCE = new UnavailableMetalCustomKernelBridge();
    private static final String REASON = "custom Metal kernel bridge unavailable";

    private UnavailableMetalCustomKernelBridge() {
    }

    @Override
    public MetalCustomKernelCapabilities capabilities() {
        return MetalCustomKernelCapabilities.unavailable(REASON);
    }

    @Override
    public MetalCustomKernelExecutable compile(MetalPartitionPlan plan) {
        return MetalCustomKernelExecutable.unavailable(REASON);
    }
}
