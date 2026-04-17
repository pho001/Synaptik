package config.runtime;

import backend.ApproxMode;
import config.backend.CpuKernelConfig;
import config.backend.KernelTuningConfig;
import config.backend.SumAccuracyMode;

import java.util.Objects;

public record RuntimeConfig(
        KernelTuningConfig kernel,
        ApproximationConfig approximation,
        BlasConfig blas,
        FusedExecutionPolicy fused
) {
    public RuntimeConfig {
        kernel = Objects.requireNonNull(kernel, "kernel cannot be null");
        approximation = approximation == null ? ApproximationConfig.defaults() : approximation;
        blas = blas == null ? BlasConfig.disabled() : blas;
        fused = fused == null ? FusedExecutionPolicy.defaultsTraining() : fused;
    }

    public RuntimeConfig(
            KernelTuningConfig kernel,
            ApproximationConfig approximation,
            BlasConfig blas
    ) {
        this(kernel, approximation, blas, FusedExecutionPolicy.defaultsTraining());
    }

    public RuntimeConfig(
            CpuKernelConfig cpuKernelConfig,
            ApproximationConfig approximation,
            BlasConfig blas
    ) {
        this(cpuKernelConfig, approximation, blas, FusedExecutionPolicy.defaultsTraining());
    }

    public RuntimeConfig(
            CpuKernelConfig cpuKernelConfig,
            ApproximationConfig approximation,
            BlasConfig blas,
            FusedExecutionPolicy fused
    ) {
        this(
                new KernelTuningConfig(
                        Objects.requireNonNull(cpuKernelConfig, "cpuKernelConfig cannot be null"),
                        KernelTuningConfig.defaultsTraining().cuda(),
                        KernelTuningConfig.defaultsTraining().opencl()
                ),
                approximation,
                blas,
                fused
        );
    }

    public static RuntimeConfig trainingDefaults() {
        return new RuntimeConfig(
                KernelTuningConfig.defaultsTraining(),
                ApproximationConfig.defaults(),
                BlasConfig.disabled(),
                FusedExecutionPolicy.defaultsTraining()
        );
    }

    public static RuntimeConfig inferenceDefaults() {
        return new RuntimeConfig(
                KernelTuningConfig.defaultsInference(),
                ApproximationConfig.defaults(),
                BlasConfig.disabled(),
                FusedExecutionPolicy.defaultsInference()
        );
    }

    public static RuntimeConfig noOptNoVecNoPar() {
        CpuKernelConfig cpuNoVecNoPar = new CpuKernelConfig(
                1,                  // loopUnrollFactor
                16, 16, 16,         // matmul tiles, tady v zásadě irelevantní bez BLAS/vector/parallel
                Integer.MAX_VALUE,  // cheapVectorMinSize => vektor se prakticky nikdy nezapne
                Integer.MAX_VALUE,  // transcendentalVectorMinSize => vektor se prakticky nikdy nezapne
                Integer.MAX_VALUE,  // reductionVectorMinSize => vektor se prakticky nikdy nezapne
                Integer.MAX_VALUE,  // cheapParallelMinSize => paralelizace se prakticky nikdy nezapne
                Integer.MAX_VALUE,  // transcendentalParallelMinSize => paralelizace se prakticky nikdy nezapne
                Integer.MAX_VALUE,  // reductionParallelMinSize => paralelizace se prakticky nikdy nezapne
                Integer.MAX_VALUE,  // contiguousMaterializeThreshold
                1,                  // lowCostTargetChunksPerWorker
                1,                  // mediumCostTargetChunksPerWorker
                1,                  // highCostTargetChunksPerWorker
                Integer.MAX_VALUE,  // minScalarChunkSize
                Integer.MAX_VALUE,  // minVectorChunkSize
                Integer.MAX_VALUE,  // minReductionChunkSize
                Integer.MAX_VALUE,  // commonPoolLowCostMaxWorkPerWorker
                1,                  // fusedAsmVectorWidth
                SumAccuracyMode.FAST,
                Integer.MAX_VALUE,  // matMulParallelMinSize => matmul paralelizace se prakticky nikdy nezapne
                config.backend.AttentionMatMulPolicy.AUTO
        );
        RuntimeConfig runtime = new RuntimeConfig(
                cpuNoVecNoPar,
                new ApproximationConfig(ApproxMode.OFF, true), // bez fast aproximací
                BlasConfig.disabled(),                         // bez BLAS
                new FusedExecutionPolicy(
                        FusedPrimaryBackend.ASM,
                        false
                )

        );
        return runtime;
    }




    public backend.runtime.RuntimeConfig toBackendRuntimeConfig() {
        return new backend.runtime.RuntimeConfig(
                kernel.cpu(),
                approximation.toBackendRuntimeConfig(),
                blas.toBackendRuntimeConfig(),
                fused
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
                BlasConfig.fromBackendRuntimeConfig(config.blasConfig()),
                config.fusedExecutionPolicy()
        );
    }
}
