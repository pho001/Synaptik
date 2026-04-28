package config.backend;

/**
 * Backend-family kernel tuning values grouped into one runtime configuration section.
 *
 * <p>The CPU section is used heavily today; CUDA and OpenCL sections preserve the same runtime profile
 * shape for accelerator backends. Instances are immutable value objects.</p>
 */
public final class KernelTuningConfig {
    private final CpuKernelConfig cpu;
    private final CudaKernelConfig cuda;
    private final OpenClKernelConfig opencl;

    /**
     * Creates grouped kernel tuning values.
     *
     * @param cpu CPU kernel tuning config
     * @param cuda CUDA kernel tuning config
     * @param opencl OpenCL kernel tuning config
     */
    public KernelTuningConfig(CpuKernelConfig cpu, CudaKernelConfig cuda, OpenClKernelConfig opencl) {
        this.cpu = cpu;
        this.cuda = cuda;
        this.opencl = opencl;
    }

    /**
     * @return CPU kernel tuning config
     */
    public CpuKernelConfig cpu() {
        return cpu;
    }

    /**
     * @return CUDA kernel tuning config
     */
    public CudaKernelConfig cuda() {
        return cuda;
    }

    /**
     * @return OpenCL kernel tuning config
     */
    public OpenClKernelConfig opencl() {
        return opencl;
    }

    /**
     * @return grouped kernel defaults for training-capable execution
     */
    public static KernelTuningConfig defaultsTraining() {
        return new KernelTuningConfig(
                CpuKernelConfig.defaultsTraining(),
                CudaKernelConfig.defaultsTraining(),
                OpenClKernelConfig.defaultsTraining()
        );
    }

    /**
     * @return grouped kernel defaults for forward-only inference
     */
    public static KernelTuningConfig defaultsInference() {
        return new KernelTuningConfig(
                CpuKernelConfig.defaultsInference(),
                CudaKernelConfig.defaultsInference(),
                OpenClKernelConfig.defaultsInference()
        );
    }
}
