package backend.accelerator.select;

import backend.ComputeBackend;
import backend.metal.bridge.MetalMpsFfmBridge;
import backend.cuda.bridge.CudaFfmBridge;

/**
 * Runtime availability probe for accelerator backends.
 */
public final class AcceleratorRuntimeAvailability {
    private AcceleratorRuntimeAvailability() {
    }

    /**
     * Returns whether the selected backend has a usable native bridge on this machine.
     */
    public static boolean isAvailable(ComputeBackend backend) {
        if (backend == null) {
            return false;
        }
        return switch (backend) {
            case GPU_METAL -> new MetalMpsFfmBridge().isAvailable();
            case GPU_CUDA -> new CudaFfmBridge().isAvailable();
            case GPU_OPENCL, CPU -> false;
        };
    }
}
