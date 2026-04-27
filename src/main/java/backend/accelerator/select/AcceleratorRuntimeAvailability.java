package backend.accelerator.select;

import backend.ComputeBackend;
import backend.metal.bridge.MetalMpsFfmBridge;
import backend.cuda.bridge.CudaFfmBridge;

public final class AcceleratorRuntimeAvailability {
    private AcceleratorRuntimeAvailability() {
    }

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
