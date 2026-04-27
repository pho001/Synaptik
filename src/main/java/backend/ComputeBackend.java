package backend;

public enum ComputeBackend {
    CPU,          // Backend for CPU computations
    GPU_CUDA,     // Backend for GPU computations using CUDA
    GPU_OPENCL,   // Backend for GPU computations using OpenCL
    GPU_METAL;    // Backend for GPU computations using Metal runtime

    /**
     * Optional: You can add methods or properties to the enum if needed.
     * For example, you might want to provide a description or some metadata.
     */

    public String getDescription() {
        switch (this) {
            case CPU:
                return "Compute backend for CPU-based computations.";
            case GPU_CUDA:
                return "Compute backend for GPU computations using CUDA.";
            case GPU_OPENCL:
                return "Compute backend for GPU computations using OpenCL.";
            case GPU_METAL:
                return "Compute backend for GPU computations using Metal runtime.";
            default:
                throw new IllegalStateException("Unexpected value: " + this);
        }
    }
}
