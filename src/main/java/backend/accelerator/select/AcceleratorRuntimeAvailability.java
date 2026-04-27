package backend.accelerator.select;

import backend.ComputeBackend;
import backend.apple.bridge.AppleMpsFfmBridge;
import backend.cuda.bridge.CudaFfmBridge;

public final class AcceleratorRuntimeAvailability {
    private AcceleratorRuntimeAvailability() {
    }

    public static boolean isAvailable(ComputeBackend backend) {
        if (backend == null) {
            return false;
        }
        return switch (backend) {
            case GPU_METAL -> new AppleMpsFfmBridge().isAvailable();
            case GPU_CUDA -> new CudaFfmBridge().isAvailable();
            case GPU_OPENCL, CPU -> false;
        };
    }
}
