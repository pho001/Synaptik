package config.runtime;

import config.backend.CpuKernelConfig;
import config.backend.KernelTuningConfig;

import java.util.Objects;

public record RuntimeConfig(
        KernelTuningConfig kernel,
        ApproximationConfig approximation,
        BlasConfig blas
) {
    public RuntimeConfig {
        kernel = Objects.requireNonNull(kernel, "kernel cannot be null");
        approximation = approximation == null ? ApproximationConfig.defaults() : approximation;
        blas = blas == null ? BlasConfig.disabled() : blas;
    }

    public RuntimeConfig(
            CpuKernelConfig cpuKernelConfig,
            ApproximationConfig approximation,
            BlasConfig blas
    ) {
        this(
                new KernelTuningConfig(
                        Objects.requireNonNull(cpuKernelConfig, "cpuKernelConfig cannot be null"),
                        KernelTuningConfig.defaultsTraining().cuda(),
                        KernelTuningConfig.defaultsTraining().opencl()
                ),
                approximation,
                blas
        );
    }

    public static RuntimeConfig trainingDefaults() {
        return new RuntimeConfig(
                KernelTuningConfig.defaultsTraining(),
                ApproximationConfig.defaults(),
                BlasConfig.disabled()
        );
    }

    public static RuntimeConfig inferenceDefaults() {
        return new RuntimeConfig(
                KernelTuningConfig.defaultsInference(),
                ApproximationConfig.defaults(),
                BlasConfig.disabled()
        );
    }

    public backend.runtime.RuntimeConfig toBackendRuntimeConfig() {
        return new backend.runtime.RuntimeConfig(
                kernel.cpu(),
                approximation.toBackendRuntimeConfig(),
                blas.toBackendRuntimeConfig()
        );
    }

    public CpuKernelConfig cpuKernelConfig() {
        return kernel.cpu();
    }

    public static RuntimeConfig fromBackendRuntimeConfig(backend.runtime.RuntimeConfig config) {
        if (config == null) {
            return trainingDefaults();
        }
        return new RuntimeConfig(
                new KernelTuningConfig(
                        config.cpuKernelConfig(),
                        KernelTuningConfig.defaultsTraining().cuda(),
                        KernelTuningConfig.defaultsTraining().opencl()
                ),
                ApproximationConfig.fromBackendRuntimeConfig(config.approximationConfig()),
                BlasConfig.fromBackendRuntimeConfig(config.blasConfig())
        );
    }
}
