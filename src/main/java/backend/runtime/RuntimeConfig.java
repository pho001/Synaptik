package backend.runtime;

import config.backend.CpuKernelConfig;
import config.runtime.FusedExecutionPolicy;

import java.util.Objects;

public record RuntimeConfig(
        CpuKernelConfig cpuKernelConfig,
        ApproximationConfig approximationConfig,
        BlasConfig blasConfig,
        FusedExecutionPolicy fusedExecutionPolicy
) {
    public RuntimeConfig {
        cpuKernelConfig = Objects.requireNonNull(cpuKernelConfig, "cpuKernelConfig cannot be null");
        approximationConfig = approximationConfig == null ? ApproximationConfig.defaults() : approximationConfig;
        blasConfig = blasConfig == null ? BlasConfig.disabled() : blasConfig;
        fusedExecutionPolicy = fusedExecutionPolicy == null ? FusedExecutionPolicy.defaultsTraining() : fusedExecutionPolicy;
    }

    public static RuntimeConfig trainingDefaults() {
        return new RuntimeConfig(
                CpuKernelConfig.defaultsTraining(),
                ApproximationConfig.defaults(),
                BlasConfig.disabled(),
                FusedExecutionPolicy.defaultsTraining()
        );
    }

    public RuntimeConfig(
            CpuKernelConfig cpuKernelConfig,
            ApproximationConfig approximationConfig,
            BlasConfig blasConfig
    ) {
        this(cpuKernelConfig, approximationConfig, blasConfig, FusedExecutionPolicy.defaultsTraining());
    }

    public static RuntimeConfig inferenceDefaults() {
        return new RuntimeConfig(
                CpuKernelConfig.defaultsInference(),
                ApproximationConfig.defaults(),
                BlasConfig.disabled(),
                FusedExecutionPolicy.defaultsInference()
        );
    }
}
