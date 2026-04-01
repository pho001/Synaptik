package backend.runtime;

import config.backend.CpuKernelConfig;

import java.util.Objects;

public record RuntimeConfig(
        CpuKernelConfig cpuKernelConfig,
        ApproximationConfig approximationConfig,
        BlasConfig blasConfig
) {
    public RuntimeConfig {
        cpuKernelConfig = Objects.requireNonNull(cpuKernelConfig, "cpuKernelConfig cannot be null");
        approximationConfig = approximationConfig == null ? ApproximationConfig.defaults() : approximationConfig;
        blasConfig = blasConfig == null ? BlasConfig.disabled() : blasConfig;
    }

    public static RuntimeConfig trainingDefaults() {
        return new RuntimeConfig(
                CpuKernelConfig.defaultsTraining(),
                ApproximationConfig.defaults(),
                BlasConfig.disabled()
        );
    }

    public static RuntimeConfig inferenceDefaults() {
        return new RuntimeConfig(
                CpuKernelConfig.defaultsInference(),
                ApproximationConfig.defaults(),
                BlasConfig.disabled()
        );
    }
}