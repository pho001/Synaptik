package config.runtime;

import backend.ApproxMode;
import backend.runtime.ExecutionMode;
import config.backend.CpuKernelConfig;
import config.backend.KernelTuningConfig;
import config.backend.SumAccuracyMode;
import config.profile.PlatformRuntimeProfileResolver;
import tensor.DataType;

import java.util.Objects;

/**
 * Runtime/backend policy used when prepared graph steps execute.
 *
 * <p>This value controls hardware-facing execution choices: CPU kernel thresholds, approximation
 * policy, BLAS and conv2d dispatch, fused execution backend, and accelerator availability. It is
 * deliberately separate from graph optimizer policy. A runnable {@link config.profile.ExecutionProfile}
 * combines this runtime config with a graph optimizer config.</p>
 *
 * <p>The record is immutable. Constructors normalize {@code null} optional components to defaults, but
 * the CPU/kernel configuration is required because execution cannot choose kernels without it.</p>
 *
 * @param kernel CPU/CUDA/OpenCL kernel tuning configuration; must not be {@code null}
 * @param approximation numerical approximation policy; {@code null} uses defaults
 * @param blas BLAS dispatch policy; {@code null} disables BLAS
 * @param conv2d conv2d GEMM dispatch policy; {@code null} derives from BLAS config
 * @param fused fused execution backend policy; {@code null} uses training defaults
 * @param accelerator accelerator backend policy; {@code null} uses training defaults
 * @param cpuStorageProfile runtime-level CPU storage policy; {@code null} uses {@link CpuStorageProfile#CPU_ARRAY}
 * @param nativeCpuFailurePolicy native CPU fallback policy; {@code null} uses
 *                               {@link NativeCpuFailurePolicy#FALLBACK_TO_ARRAY}
 * @param deviceTransferPolicy host/device transfer fallback policy; {@code null} uses
 *                             {@link DeviceTransferPolicy#ALLOW_ARRAY_BRIDGE}
 * @param nativeCpuMemory native CPU allocation/pooling policy; {@code null} disables pooling
 */
public record RuntimeConfig(
        KernelTuningConfig kernel,
        ApproximationConfig approximation,
        BlasConfig blas,
        Conv2dConfig conv2d,
        FusedExecutionPolicy fused,
        AcceleratorConfig accelerator,
        CpuStorageProfile cpuStorageProfile,
        NativeCpuFailurePolicy nativeCpuFailurePolicy,
        DeviceTransferPolicy deviceTransferPolicy,
        NativeCpuMemoryConfig nativeCpuMemory
) {
    public RuntimeConfig {
        kernel = Objects.requireNonNull(kernel, "kernel cannot be null");
        approximation = approximation == null ? ApproximationConfig.defaults() : approximation;
        blas = blas == null ? BlasConfig.disabled() : blas;
        conv2d = conv2d == null ? Conv2dConfig.fromBlasConfig(blas) : conv2d;
        fused = fused == null ? FusedExecutionPolicy.defaultsTraining() : fused;
        accelerator = accelerator == null ? AcceleratorConfig.defaultsTraining() : accelerator;
        cpuStorageProfile = cpuStorageProfile == null ? CpuStorageProfile.CPU_ARRAY : cpuStorageProfile;
        nativeCpuFailurePolicy = nativeCpuFailurePolicy == null
                ? NativeCpuFailurePolicy.FALLBACK_TO_ARRAY
                : nativeCpuFailurePolicy;
        deviceTransferPolicy = deviceTransferPolicy == null
                ? DeviceTransferPolicy.ALLOW_ARRAY_BRIDGE
                : deviceTransferPolicy;
        nativeCpuMemory = nativeCpuMemory == null ? NativeCpuMemoryConfig.disabled() : nativeCpuMemory;
    }

    public RuntimeConfig(
            KernelTuningConfig kernel,
            ApproximationConfig approximation,
            BlasConfig blas,
            Conv2dConfig conv2d,
            FusedExecutionPolicy fused,
            AcceleratorConfig accelerator,
            CpuStorageProfile cpuStorageProfile,
            NativeCpuFailurePolicy nativeCpuFailurePolicy,
            NativeCpuMemoryConfig nativeCpuMemory
    ) {
        this(
                kernel,
                approximation,
                blas,
                conv2d,
                fused,
                accelerator,
                cpuStorageProfile,
                nativeCpuFailurePolicy,
                DeviceTransferPolicy.ALLOW_ARRAY_BRIDGE,
                nativeCpuMemory
        );
    }

    public RuntimeConfig(
            KernelTuningConfig kernel,
            ApproximationConfig approximation,
            BlasConfig blas,
            Conv2dConfig conv2d,
            FusedExecutionPolicy fused,
            AcceleratorConfig accelerator,
            CpuStorageProfile cpuStorageProfile,
            NativeCpuFailurePolicy nativeCpuFailurePolicy
    ) {
        this(
                kernel,
                approximation,
                blas,
                conv2d,
                fused,
                accelerator,
                cpuStorageProfile,
                nativeCpuFailurePolicy,
                DeviceTransferPolicy.ALLOW_ARRAY_BRIDGE,
                NativeCpuMemoryConfig.disabled()
        );
    }

    public RuntimeConfig(
            KernelTuningConfig kernel,
            ApproximationConfig approximation,
            BlasConfig blas,
            Conv2dConfig conv2d,
            FusedExecutionPolicy fused,
            AcceleratorConfig accelerator
    ) {
        this(
                kernel,
                approximation,
                blas,
                conv2d,
                fused,
                accelerator,
                CpuStorageProfile.CPU_ARRAY,
                NativeCpuFailurePolicy.FALLBACK_TO_ARRAY
        );
    }

    /**
     * Creates a runtime config using conv2d, fused, and accelerator defaults.
     *
     * @param kernel kernel tuning configuration
     * @param approximation approximation policy
     * @param blas BLAS policy
     */
    public RuntimeConfig(
            KernelTuningConfig kernel,
            ApproximationConfig approximation,
            BlasConfig blas
    ) {
        this(
                kernel,
                approximation,
                blas,
                Conv2dConfig.fromBlasConfig(blas),
                FusedExecutionPolicy.defaultsTraining(),
                AcceleratorConfig.defaultsTraining()
        );
    }

    public RuntimeConfig(
            KernelTuningConfig kernel,
            ApproximationConfig approximation,
            BlasConfig blas,
            FusedExecutionPolicy fused
    ) {
        this(kernel, approximation, blas, Conv2dConfig.fromBlasConfig(blas), fused, AcceleratorConfig.defaultsTraining());
    }

    public RuntimeConfig(
            KernelTuningConfig kernel,
            ApproximationConfig approximation,
            BlasConfig blas,
            FusedExecutionPolicy fused,
            AcceleratorConfig accelerator
    ) {
        this(kernel, approximation, blas, Conv2dConfig.fromBlasConfig(blas), fused, accelerator);
    }

    public RuntimeConfig(
            KernelTuningConfig kernel,
            ApproximationConfig approximation,
            BlasConfig blas,
            Conv2dConfig conv2d,
            FusedExecutionPolicy fused
    ) {
        this(kernel, approximation, blas, conv2d, fused, AcceleratorConfig.defaultsTraining());
    }

    public RuntimeConfig(
            CpuKernelConfig cpuKernelConfig,
            ApproximationConfig approximation,
            BlasConfig blas
    ) {
        this(
                cpuKernelConfig,
                approximation,
                blas,
                Conv2dConfig.fromBlasConfig(blas),
                FusedExecutionPolicy.defaultsTraining(),
                AcceleratorConfig.defaultsTraining()
        );
    }

    public RuntimeConfig(
            CpuKernelConfig cpuKernelConfig,
            ApproximationConfig approximation,
            BlasConfig blas,
            FusedExecutionPolicy fused
    ) {
        this(
                cpuKernelConfig,
                approximation,
                blas,
                Conv2dConfig.fromBlasConfig(blas),
                fused,
                AcceleratorConfig.defaultsTraining()
        );
    }

    public RuntimeConfig(
            CpuKernelConfig cpuKernelConfig,
            ApproximationConfig approximation,
            BlasConfig blas,
            FusedExecutionPolicy fused,
            AcceleratorConfig accelerator
    ) {
        this(
                cpuKernelConfig,
                approximation,
                blas,
                Conv2dConfig.fromBlasConfig(blas),
                fused,
                accelerator
        );
    }

    public RuntimeConfig(
            CpuKernelConfig cpuKernelConfig,
            ApproximationConfig approximation,
            BlasConfig blas,
            Conv2dConfig conv2d,
            FusedExecutionPolicy fused
    ) {
        this(cpuKernelConfig, approximation, blas, conv2d, fused, AcceleratorConfig.defaultsTraining());
    }

    public RuntimeConfig(
            CpuKernelConfig cpuKernelConfig,
            ApproximationConfig approximation,
            BlasConfig blas,
            Conv2dConfig conv2d,
            FusedExecutionPolicy fused,
            AcceleratorConfig accelerator
    ) {
        this(
                new KernelTuningConfig(
                        Objects.requireNonNull(cpuKernelConfig, "cpuKernelConfig cannot be null"),
                        KernelTuningConfig.defaultsTraining().cuda(),
                        KernelTuningConfig.defaultsTraining().opencl()
                ),
                approximation,
                blas,
                conv2d,
                fused,
                accelerator
        );
    }

    /**
     * Returns the default runtime policy for training-capable execution.
     *
     * @return runtime defaults for forward/backward mode
     */
    public static RuntimeConfig trainingDefaults() {
        return new RuntimeConfig(
                KernelTuningConfig.defaultsTraining(),
                ApproximationConfig.defaults(),
                BlasConfig.disabled(),
                Conv2dConfig.disabled(),
                FusedExecutionPolicy.defaultsTraining(),
                AcceleratorConfig.defaultsTraining()
        );
    }

    /**
     * Returns training defaults, preferring a compatible calibrated platform runtime profile when one is available.
     *
     * <p>Lookup is dtype-aware because calibration thresholds, BLAS policy, and materialization thresholds can differ
     * by storage/compute type. If no compatible profile exists under the configured profile roots or bundled resources,
     * this method returns {@link #trainingDefaults()}.</p>
     *
     * @param dataType graph root dtype
     * @return calibrated training runtime when present, otherwise hardcoded training defaults
     */
    public static RuntimeConfig trainingDefaults(DataType dataType) {
        RuntimeConfig fallback = trainingDefaults();
        return PlatformRuntimeProfileResolver.resolveRuntimeConfig(
                dataType,
                ExecutionMode.FORWARD_BACKWARD,
                fallback
        );
    }

    /**
     * Returns the default runtime policy for forward-only inference execution.
     *
     * @return runtime defaults for inference mode
     */
    public static RuntimeConfig inferenceDefaults() {
        return new RuntimeConfig(
                KernelTuningConfig.defaultsInference(),
                ApproximationConfig.defaults(),
                BlasConfig.disabled(),
                Conv2dConfig.disabled(),
                FusedExecutionPolicy.defaultsInference(),
                AcceleratorConfig.defaultsInference()
        );
    }

    /**
     * Returns inference defaults, preferring a compatible calibrated platform runtime profile when one is available.
     *
     * @param dataType graph root dtype
     * @return calibrated inference runtime when present, otherwise hardcoded inference defaults
     */
    public static RuntimeConfig inferenceDefaults(DataType dataType) {
        RuntimeConfig fallback = inferenceDefaults();
        return PlatformRuntimeProfileResolver.resolveRuntimeConfig(
                dataType,
                ExecutionMode.FORWARD,
                fallback
        );
    }

    /**
     * Returns a deliberately conservative runtime baseline with no BLAS, no vector path, and no
     * parallel dispatch under practical workload sizes.
     *
     * <p>This is used for benchmark comparisons where the graph optimizer baseline also disables graph
     * optimization. The thresholds are set extremely high so vector/parallel branches are effectively
     * unreachable for normal tests, while scalar execution remains functional.</p>
     *
     * @return runtime baseline for performance comparison
     */
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
                ),
                AcceleratorConfig.disabled()

        );
        return runtime;
    }

    /**
     * Returns the CPU kernel tuning section.
     *
     * @return CPU kernel configuration from {@link #kernel()}
     */
    public CpuKernelConfig cpuKernelConfig() {
        return kernel.cpu();
    }

    /**
     * Returns a copy with a different accelerator policy.
     *
     * @param newAccelerator replacement accelerator config; {@code null} uses training defaults
     * @return runtime config with the same kernel, approximation, BLAS, conv2d, and fused settings
     */
    public RuntimeConfig withAccelerator(AcceleratorConfig newAccelerator) {
        return new RuntimeConfig(
                kernel,
                approximation,
                blas,
                conv2d,
                fused,
                newAccelerator,
                cpuStorageProfile,
                nativeCpuFailurePolicy,
                deviceTransferPolicy,
                nativeCpuMemory
        );
    }

    /**
     * Returns a copy with a different CPU storage profile.
     *
     * @param newCpuStorageProfile replacement CPU storage policy; {@code null} uses {@code CPU_ARRAY}
     * @return runtime config with the same kernel, approximation, BLAS, conv2d, fused, accelerator, and failure policy
     */
    public RuntimeConfig withCpuStorageProfile(CpuStorageProfile newCpuStorageProfile) {
        return new RuntimeConfig(
                kernel,
                approximation,
                blas,
                conv2d,
                fused,
                accelerator,
                newCpuStorageProfile,
                nativeCpuFailurePolicy,
                deviceTransferPolicy,
                nativeCpuMemory
        );
    }

    /**
     * Returns a copy with a different native CPU failure policy.
     *
     * @param newNativeCpuFailurePolicy replacement native failure policy; {@code null} uses fallback-to-array
     * @return runtime config with the same kernel, approximation, BLAS, conv2d, fused, accelerator, and storage policy
     */
    public RuntimeConfig withNativeCpuFailurePolicy(NativeCpuFailurePolicy newNativeCpuFailurePolicy) {
        return new RuntimeConfig(
                kernel,
                approximation,
                blas,
                conv2d,
                fused,
                accelerator,
                cpuStorageProfile,
                newNativeCpuFailurePolicy,
                deviceTransferPolicy,
                nativeCpuMemory
        );
    }

    /**
     * Returns a copy with a different host/device transfer policy.
     *
     * @param newDeviceTransferPolicy replacement transfer policy; {@code null} allows array bridge fallback
     * @return runtime config with the same execution policy and updated transfer policy
     */
    public RuntimeConfig withDeviceTransferPolicy(DeviceTransferPolicy newDeviceTransferPolicy) {
        return new RuntimeConfig(
                kernel,
                approximation,
                blas,
                conv2d,
                fused,
                accelerator,
                cpuStorageProfile,
                nativeCpuFailurePolicy,
                newDeviceTransferPolicy,
                nativeCpuMemory
        );
    }

    /**
     * Returns a copy with a different native CPU memory policy.
     *
     * @param newNativeCpuMemory replacement native memory policy; {@code null} disables pooling
     * @return runtime config with the same execution policy and updated native memory config
     */
    public RuntimeConfig withNativeCpuMemory(NativeCpuMemoryConfig newNativeCpuMemory) {
        return new RuntimeConfig(
                kernel,
                approximation,
                blas,
                conv2d,
                fused,
                accelerator,
                cpuStorageProfile,
                nativeCpuFailurePolicy,
                deviceTransferPolicy,
                newNativeCpuMemory
        );
    }
}
