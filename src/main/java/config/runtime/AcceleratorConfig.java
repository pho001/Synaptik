package config.runtime;

import backend.contract.ComputeBackend;

import java.util.Objects;

/**
 * Runtime accelerator policy for CUDA, OpenCL, and Metal backends.
 *
 * <p>Graph/backend selection reads this config when deciding whether an accelerator candidate is
 * eligible. A disabled backend is never selected through normal policy. CPU execution is not configured
 * here and always maps to a disabled accelerator backend config.</p>
 *
 * @param cuda CUDA backend policy; {@code null} uses defaults
 * @param opencl OpenCL backend policy; {@code null} uses defaults
 * @param metal Metal backend policy; {@code null} uses defaults
 */
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

    /**
     * @return default accelerator policy for all accelerator backends
     */
    public static AcceleratorConfig defaults() {
        return new AcceleratorConfig(
                AcceleratorBackendConfig.defaults(),
                AcceleratorBackendConfig.defaults(),
                AcceleratorBackendConfig.defaults()
        );
    }

    /**
     * @return default accelerator policy for training-capable execution
     */
    public static AcceleratorConfig defaultsTraining() {
        return defaults();
    }

    /**
     * @return default accelerator policy for forward-only inference
     */
    public static AcceleratorConfig defaultsInference() {
        return defaults();
    }

    /**
     * @return config with all accelerator backends disabled
     */
    public static AcceleratorConfig disabled() {
        return new AcceleratorConfig(
                AcceleratorBackendConfig.disabled(),
                AcceleratorBackendConfig.disabled(),
                AcceleratorBackendConfig.disabled()
        );
    }

    /**
     * Returns the accelerator policy for a backend enum.
     *
     * @param backend backend to inspect; must not be {@code null}
     * @return backend-specific accelerator policy, or a disabled policy for CPU
     */
    public AcceleratorBackendConfig forBackend(ComputeBackend backend) {
        return switch (backend) {
            case GPU_CUDA -> cuda;
            case GPU_OPENCL -> opencl;
            case GPU_METAL -> metal;
            case CPU -> AcceleratorBackendConfig.disabled();
        };
    }

    /**
     * Returns a copy with a replacement CUDA policy.
     *
     * @param newCuda replacement CUDA policy; {@code null} uses defaults
     * @return updated config
     */
    public AcceleratorConfig withCuda(AcceleratorBackendConfig newCuda) {
        return new AcceleratorConfig(newCuda, opencl, metal);
    }

    /**
     * Returns a copy with a replacement OpenCL policy.
     *
     * @param newOpencl replacement OpenCL policy; {@code null} uses defaults
     * @return updated config
     */
    public AcceleratorConfig withOpencl(AcceleratorBackendConfig newOpencl) {
        return new AcceleratorConfig(cuda, newOpencl, metal);
    }

    /**
     * Returns a copy with a replacement Metal policy.
     *
     * @param newMetal replacement Metal policy; {@code null} uses defaults
     * @return updated config
     */
    public AcceleratorConfig withMetal(AcceleratorBackendConfig newMetal) {
        return new AcceleratorConfig(cuda, opencl, newMetal);
    }
}
