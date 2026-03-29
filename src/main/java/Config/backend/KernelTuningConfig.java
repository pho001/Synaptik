package Config.backend;

public final class KernelTuningConfig {
    private final CpuKernelConfig cpu;
    private final CudaKernelConfig cuda;
    private final OpenClKernelConfig opencl;

    public KernelTuningConfig(CpuKernelConfig cpu, CudaKernelConfig cuda, OpenClKernelConfig opencl) {
        this.cpu = cpu;
        this.cuda = cuda;
        this.opencl = opencl;
    }

    public CpuKernelConfig cpu() {
        return cpu;
    }

    public CudaKernelConfig cuda() {
        return cuda;
    }

    public OpenClKernelConfig opencl() {
        return opencl;
    }

    public static KernelTuningConfig defaultsTraining() {
        return new KernelTuningConfig(
                CpuKernelConfig.defaultsTraining(),
                CudaKernelConfig.defaultsTraining(),
                OpenClKernelConfig.defaultsTraining()
        );
    }

    public static KernelTuningConfig defaultsInference() {
        return new KernelTuningConfig(
                CpuKernelConfig.defaultsInference(),
                CudaKernelConfig.defaultsInference(),
                OpenClKernelConfig.defaultsInference()
        );
    }
}
