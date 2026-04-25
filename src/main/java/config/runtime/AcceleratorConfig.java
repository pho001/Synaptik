package config.runtime;

import backend.ComputeBackend;

import java.util.Objects;

public record AcceleratorConfig(
        AcceleratorBackendConfig cuda,
        AcceleratorBackendConfig opencl,
        AcceleratorBackendConfig metal
) {
    public AcceleratorConfig {
        cuda = Objects.requireNonNullElse(cuda, AcceleratorBackendConfig.defaults());
        opencl = Objects.requireNonNullElse(opencl, AcceleratorBackendConfig.defaults());
        metal = Objects.requireNonNullElse(metal, AcceleratorBackendConfig.defaults());
    }

    public static AcceleratorConfig defaults() {
        return new AcceleratorConfig(
                AcceleratorBackendConfig.defaults(),
                AcceleratorBackendConfig.defaults(),
                AcceleratorBackendConfig.defaults()
        );
    }

    public static AcceleratorConfig defaultsTraining() {
        return defaults();
    }

    public static AcceleratorConfig defaultsInference() {
        return defaults();
    }

    public static AcceleratorConfig disabled() {
        return new AcceleratorConfig(
                AcceleratorBackendConfig.disabled(),
                AcceleratorBackendConfig.disabled(),
                AcceleratorBackendConfig.disabled()
        );
    }

    public AcceleratorBackendConfig forBackend(ComputeBackend backend) {
        return switch (backend) {
            case GPU_CUDA -> cuda;
            case GPU_OPENCL -> opencl;
            case GPU_METAL -> metal;
            case CPU -> AcceleratorBackendConfig.disabled();
        };
    }

    public AcceleratorConfig withCuda(AcceleratorBackendConfig newCuda) {
        return new AcceleratorConfig(newCuda, opencl, metal);
    }

    public AcceleratorConfig withOpencl(AcceleratorBackendConfig newOpencl) {
        return new AcceleratorConfig(cuda, newOpencl, metal);
    }

    public AcceleratorConfig withMetal(AcceleratorBackendConfig newMetal) {
        return new AcceleratorConfig(cuda, opencl, newMetal);
    }
}
