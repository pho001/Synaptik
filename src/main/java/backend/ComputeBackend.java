package backend;

/**
 * Backend family selected for a compiled execution node or partition.
 *
 * <p>The enum is used in optimizer partition planning, backend selection, prepare-time metadata, and
 * runtime dispatch. GPU values describe the intended accelerator family; runtime availability is checked
 * separately through backend selection and accelerator configuration.</p>
 */
public enum ComputeBackend {
    /**
     * CPU execution backend.
     */
    CPU,
    /**
     * CUDA GPU backend.
     */
    GPU_CUDA,
    /**
     * OpenCL GPU backend.
     */
    GPU_OPENCL,
    /**
     * Metal GPU backend.
     */
    GPU_METAL;

    /**
     * Returns a human-readable backend description.
     *
     * @return short description suitable for diagnostics
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
