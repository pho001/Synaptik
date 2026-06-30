package config.profile;

import backend.blas.BlasProvider;
import config.backend.AttentionMatMulPolicy;
import config.backend.CpuMatMulMicroKernel;
import config.backend.CpuKernelConfig;
import config.backend.KernelTuningConfig;
import config.runtime.ApproximationConfig;
import config.runtime.BlasConfig;
import config.runtime.CpuStorageProfile;
import config.runtime.DeviceTransferPolicy;
import config.runtime.FusedExecutionPolicy;
import config.runtime.NativeCpuFailurePolicy;
import config.runtime.RuntimeConfig;
import tensor.DataType;

import java.time.OffsetDateTime;
import java.util.Objects;

/**
 * Calibrated hardware/runtime profile for one platform, dtype, and execution mode.
 *
 * <p>This profile is the output of platform calibration. It stores runtime thresholds and backend
 * dispatch choices that depend on the current machine, such as matmul/conv2d BLAS thresholds,
 * fused-kernel vector and parallel thresholds, scheduler chunking, materialization thresholds, numeric
 * approximation policy, and accelerator availability policy. Graph autotune consumes this object as a
 * frozen runtime input and varies only {@link GraphExecutionPolicy}.</p>
 *
 * <p>The profile can be converted back to {@link RuntimeConfig} before execution. The conversion keeps
 * graph policy separate: it does not decide optimizer stage order, CSE mode, partitioning, or memory
 * planning.</p>
 *
 * @param metadata platform id, hardware key, schema versions, dtype, and execution mode
 * @param matmul calibrated matmul and attention-matmul runtime settings
 * @param conv2d legacy calibrated conv2d dispatch settings; {@code null} derives defaults from matmul
 * @param fused calibrated fused elementwise dispatch and generated-ASM vector width settings
 * @param elementwiseDispatch calibrated non-fused elementwise dispatch thresholds
 * @param reduction calibrated reduction and attention-reduction thresholds
 * @param scheduler calibrated CPU chunking policy
 * @param materialization calibrated layout/materialization thresholds
 * @param numerics numerical approximation policy copied into runtime config
 * @param accelerator accelerator backend selection policy; {@code null} uses defaults
 * @param cpuStorageProfile runtime-level CPU storage policy; {@code null} uses {@code CPU_ARRAY}
 * @param nativeCpuFailurePolicy native CPU fallback policy; {@code null} uses {@code FALLBACK_TO_ARRAY}
 * @param deviceTransferPolicy host/device transfer fallback policy; {@code null} uses {@code ALLOW_ARRAY_BRIDGE}
 */
public record PlatformRuntimeProfile(
        PlatformProfileMetadata metadata,
        MatmulPlatformProfile matmul,
        Conv2dPlatformProfile conv2d,
        FusedPlatformProfile fused,
        ElementwiseDispatchPlatformProfile elementwiseDispatch,
        ReductionPlatformProfile reduction,
        SchedulerPlatformProfile scheduler,
        MaterializationPlatformProfile materialization,
        NumericsPlatformProfile numerics,
        AcceleratorPlatformProfile accelerator,
        CpuStorageProfile cpuStorageProfile,
        NativeCpuFailurePolicy nativeCpuFailurePolicy,
        DeviceTransferPolicy deviceTransferPolicy
) {
    public PlatformRuntimeProfile {
        Objects.requireNonNull(metadata, "metadata cannot be null");
        Objects.requireNonNull(matmul, "matmul cannot be null");
        conv2d = conv2d == null ? Conv2dPlatformProfile.fromMatmul(matmul) : conv2d;
        Objects.requireNonNull(fused, "fused cannot be null");
        Objects.requireNonNull(elementwiseDispatch, "elementwiseDispatch cannot be null");
        Objects.requireNonNull(reduction, "reduction cannot be null");
        Objects.requireNonNull(scheduler, "scheduler cannot be null");
        Objects.requireNonNull(materialization, "materialization cannot be null");
        Objects.requireNonNull(numerics, "numerics cannot be null");
        accelerator = accelerator == null ? AcceleratorPlatformProfile.defaults() : accelerator;
        cpuStorageProfile = cpuStorageProfile == null ? CpuStorageProfile.CPU_ARRAY : cpuStorageProfile;
        nativeCpuFailurePolicy = nativeCpuFailurePolicy == null
                ? NativeCpuFailurePolicy.FALLBACK_TO_ARRAY
                : nativeCpuFailurePolicy;
        deviceTransferPolicy = deviceTransferPolicy == null
                ? DeviceTransferPolicy.ALLOW_ARRAY_BRIDGE
                : deviceTransferPolicy;
    }

    public PlatformRuntimeProfile(
            PlatformProfileMetadata metadata,
            MatmulPlatformProfile matmul,
            Conv2dPlatformProfile conv2d,
            FusedPlatformProfile fused,
            ElementwiseDispatchPlatformProfile elementwiseDispatch,
            ReductionPlatformProfile reduction,
            SchedulerPlatformProfile scheduler,
            MaterializationPlatformProfile materialization,
            NumericsPlatformProfile numerics,
            AcceleratorPlatformProfile accelerator,
            CpuStorageProfile cpuStorageProfile,
            NativeCpuFailurePolicy nativeCpuFailurePolicy
    ) {
        this(
                metadata,
                matmul,
                conv2d,
                fused,
                elementwiseDispatch,
                reduction,
                scheduler,
                materialization,
                numerics,
                accelerator,
                cpuStorageProfile,
                nativeCpuFailurePolicy,
                DeviceTransferPolicy.ALLOW_ARRAY_BRIDGE
        );
    }

    public PlatformRuntimeProfile(
            PlatformProfileMetadata metadata,
            MatmulPlatformProfile matmul,
            Conv2dPlatformProfile conv2d,
            FusedPlatformProfile fused,
            ElementwiseDispatchPlatformProfile elementwiseDispatch,
            ReductionPlatformProfile reduction,
            SchedulerPlatformProfile scheduler,
            MaterializationPlatformProfile materialization,
            NumericsPlatformProfile numerics,
            AcceleratorPlatformProfile accelerator
    ) {
        this(
                metadata,
                matmul,
                conv2d,
                fused,
                elementwiseDispatch,
                reduction,
                scheduler,
                materialization,
                numerics,
                accelerator,
                CpuStorageProfile.CPU_ARRAY,
                NativeCpuFailurePolicy.FALLBACK_TO_ARRAY,
                DeviceTransferPolicy.ALLOW_ARRAY_BRIDGE
        );
    }

    /**
     * Creates a platform profile with default accelerator policy.
     *
     * @param metadata platform metadata
     * @param matmul matmul runtime settings
     * @param conv2d conv2d runtime settings
     * @param fused fused runtime settings
     * @param elementwiseDispatch non-fused elementwise dispatch settings
     * @param reduction reduction settings
     * @param scheduler scheduling settings
     * @param materialization materialization settings
     * @param numerics numerical approximation policy
     */
    public PlatformRuntimeProfile(
            PlatformProfileMetadata metadata,
            MatmulPlatformProfile matmul,
            Conv2dPlatformProfile conv2d,
            FusedPlatformProfile fused,
            ElementwiseDispatchPlatformProfile elementwiseDispatch,
            ReductionPlatformProfile reduction,
            SchedulerPlatformProfile scheduler,
            MaterializationPlatformProfile materialization,
            NumericsPlatformProfile numerics
    ) {
        this(
                metadata,
                matmul,
                conv2d,
                fused,
                elementwiseDispatch,
                reduction,
                scheduler,
                materialization,
                numerics,
                AcceleratorPlatformProfile.defaults()
        );
    }

    /**
     * Creates a platform profile that derives conv2d settings from matmul settings.
     *
     * <p>This overload exists for older profile assembly paths. New calibration code should prefer the
     * overload that supplies an explicit {@link Conv2dPlatformProfile}.</p>
     *
     * @param metadata platform metadata
     * @param matmul matmul runtime settings
     * @param fused fused runtime settings
     * @param elementwiseDispatch non-fused elementwise dispatch settings
     * @param reduction reduction settings
     * @param scheduler scheduling settings
     * @param materialization materialization settings
     * @param numerics numerical approximation policy
     */
    public PlatformRuntimeProfile(
            PlatformProfileMetadata metadata,
            MatmulPlatformProfile matmul,
            FusedPlatformProfile fused,
            ElementwiseDispatchPlatformProfile elementwiseDispatch,
            ReductionPlatformProfile reduction,
            SchedulerPlatformProfile scheduler,
            MaterializationPlatformProfile materialization,
            NumericsPlatformProfile numerics
    ) {
        this(
                metadata,
                matmul,
                Conv2dPlatformProfile.fromMatmul(matmul),
                fused,
                elementwiseDispatch,
                reduction,
                scheduler,
                materialization,
                numerics,
                AcceleratorPlatformProfile.defaults()
        );
    }

    /**
     * Extracts a platform runtime profile from a complete execution profile.
     *
     * <p>This is used as a seed when no persisted calibration profile exists. It copies runtime
     * settings from {@code profile.runtime()} and records the supplied platform metadata. The resulting
     * value is a runtime profile only; graph optimizer settings from the profile are not stored here.</p>
     *
     * @param platformProfileId stable platform profile id, usually derived from hardware fingerprint
     * @param hardwareKey hardware fingerprint key
     * @param calibrationPreset name of the preset or seed source
     * @param profile source execution profile; must not be {@code null}
     * @return runtime-only platform profile derived from {@code profile.runtime()}
     */
    public static PlatformRuntimeProfile fromExecutionProfile(
            String platformProfileId,
            String hardwareKey,
            String calibrationPreset,
            ExecutionProfile profile
    ) {
        Objects.requireNonNull(profile, "profile cannot be null");
        CpuKernelConfig cpu = profile.runtime().kernel().cpu();
        PlatformProfileMetadata metadata = new PlatformProfileMetadata(
                platformProfileId,
                hardwareKey,
                "dev",
                "1",
                "1",
                OffsetDateTime.now().toString(),
                calibrationPreset,
                profile.dataType(),
                profile.mode()
        );
        return new PlatformRuntimeProfile(
                metadata,
                new MatmulPlatformProfile(
                        profile.runtime().blas().provider(),
                        profile.runtime().blas().matmulMinWork(),
                        profile.runtime().blas().threads(),
                        profile.runtime().blas().openBlasArrayCopyThreads(),
                        profile.runtime().blas().openBlasNativeSegmentThreads(),
                        profile.runtime().blas().f32RequireMgeK(),
                        profile.runtime().blas().f32MaxNOverK(),
                        profile.runtime().blas().f32WideRequireMgeK(),
                        profile.runtime().blas().f32WideMaxNOverK(),
                        profile.runtime().blas().storageMode(),
                        cpu.loopUnrollFactor(),
                        cpu.matMulTileM(),
                        cpu.matMulTileN(),
                        cpu.matMulTileK(),
                        cpu.attentionMatMulTileM(),
                        cpu.attentionMatMulTileN(),
                        cpu.attentionMatMulTileK(),
                        cpu.matMulParallelMinSize(),
                        cpu.matMulMicroKernel(),
                        cpu.attentionMatMulMicroKernel()
                ),
                new Conv2dPlatformProfile(
                        profile.runtime().conv2d().provider(),
                        profile.runtime().conv2d().f64MinWork(),
                        profile.runtime().conv2d().f32MinWork(),
                        profile.runtime().conv2d().f32RequireMgeK(),
                        profile.runtime().conv2d().f32MaxNOverK(),
                        profile.runtime().conv2d().bf16MinWork(),
                        profile.runtime().conv2d().bf16RequireMgeK(),
                        profile.runtime().conv2d().bf16MaxNOverK()
                ),
                new FusedPlatformProfile(
                        cpu.fusedCheapVectorMinSize(),
                        cpu.fusedTranscendentalVectorMinSize(),
                        cpu.fusedCheapParallelMinSize(),
                        cpu.fusedTranscendentalParallelMinSize(),
                        cpu.fusedAsmVectorWidth()
                ),
                new ElementwiseDispatchPlatformProfile(
                        cpu.cheapVectorMinSize(),
                        cpu.nativeF32CheapVectorMinSize(),
                        cpu.nativeF64CheapVectorMinSize(),
                        cpu.transcendentalVectorMinSize(),
                        cpu.cheapParallelMinSize(),
                        cpu.transcendentalParallelMinSize()
                ),
                new ReductionPlatformProfile(
                        cpu.reductionVectorMinSize(),
                        cpu.reductionParallelMinSize(),
                        cpu.attentionVectorMinSize(),
                        cpu.attentionParallelMinSize(),
                        cpu.sumAccuracyMode()
                ),
                new SchedulerPlatformProfile(
                        cpu.lowCostTargetChunksPerWorker(),
                        cpu.mediumCostTargetChunksPerWorker(),
                        cpu.highCostTargetChunksPerWorker(),
                        cpu.minScalarChunkSize(),
                        cpu.minVectorChunkSize(),
                        cpu.minReductionChunkSize(),
                        cpu.commonPoolLowCostMaxWorkPerWorker()
                ),
                new MaterializationPlatformProfile(
                        cpu.contiguousMaterializeThreshold(),
                        cpu.cheapF64MaterializeThreshold(),
                        cpu.cheapF32MaterializeThreshold(),
                        cpu.cheapBF16MaterializeThreshold(),
                        cpu.whereMaterializeThreshold()
                ),
                new NumericsPlatformProfile(
                        profile.runtime().approximation().approxMode(),
                        profile.runtime().approximation().forceExactTranscendentals()
                ),
                AcceleratorPlatformProfile.fromRuntimeConfig(profile.runtime().accelerator()),
                profile.runtime().cpuStorageProfile(),
                profile.runtime().nativeCpuFailurePolicy(),
                profile.runtime().deviceTransferPolicy()
        );
    }

    /**
     * Converts this calibrated profile into the runtime configuration consumed by graph execution.
     *
     * <p>The returned config contains backend thresholds, BLAS/conv2d dispatch settings, fused runtime
     * policy, numerical approximation policy, and accelerator settings. It does not contain graph
     * optimizer policy; combine it with a {@link GraphExecutionPolicy} through an
     * {@link ExecutionProfile} when building runnable candidates.</p>
     *
     * @return runtime configuration equivalent to this platform profile
     */
    public RuntimeConfig toRuntimeConfig() {
        CpuKernelConfig cpu = new CpuKernelConfig(
                matmul.loopUnrollFactor(),
                matmul.matMulTileM(),
                matmul.matMulTileN(),
                matmul.matMulTileK(),
                elementwiseDispatch.cheapVectorMinSize(),
                elementwiseDispatch.transcendentalVectorMinSize(),
                fused.fusedCheapVectorMinSize(),
                fused.fusedTranscendentalVectorMinSize(),
                reduction.reductionVectorMinSize(),
                reduction.attentionVectorMinSize(),
                elementwiseDispatch.cheapParallelMinSize(),
                elementwiseDispatch.transcendentalParallelMinSize(),
                fused.fusedCheapParallelMinSize(),
                fused.fusedTranscendentalParallelMinSize(),
                reduction.reductionParallelMinSize(),
                reduction.attentionParallelMinSize(),
                materialization.contiguousMaterializeThreshold(),
                materialization.cheapF64MaterializeThreshold(),
                materialization.cheapF32MaterializeThreshold(),
                materialization.cheapBF16MaterializeThreshold(),
                materialization.whereMaterializeThreshold(),
                scheduler.lowCostTargetChunksPerWorker(),
                scheduler.mediumCostTargetChunksPerWorker(),
                scheduler.highCostTargetChunksPerWorker(),
                scheduler.minScalarChunkSize(),
                scheduler.minVectorChunkSize(),
                scheduler.minReductionChunkSize(),
                scheduler.commonPoolLowCostMaxWorkPerWorker(),
                fused.fusedAsmVectorWidth(),
                reduction.sumAccuracyMode(),
                matmul.matMulParallelMinSize(),
                AttentionMatMulPolicy.AUTO,
                matmul.matMulMicroKernel() == null ? CpuMatMulMicroKernel.AUTO : matmul.matMulMicroKernel(),
                matmul.attentionMatMulMicroKernel() == null ? matmul.matMulMicroKernel() : matmul.attentionMatMulMicroKernel(),
                matmul.attentionMatMulTileM(),
                matmul.attentionMatMulTileN(),
                matmul.attentionMatMulTileK(),
                elementwiseDispatch.nativeF32CheapVectorMinSize(),
                elementwiseDispatch.nativeF64CheapVectorMinSize()
        );
        return new RuntimeConfig(
                new KernelTuningConfig(
                        cpu,
                        KernelTuningConfig.defaultsTraining().cuda(),
                        KernelTuningConfig.defaultsTraining().opencl()
                ),
                new ApproximationConfig(numerics.approxMode(), numerics.forceExactTranscendentals()),
                new BlasConfig(
                        matmul.blasProvider(),
                        matmul.blasMatmulMinWork(),
                        matmul.f32RequireMgeK(),
                        matmul.f32MaxNOverK(),
                        matmul.f32WideRequireMgeK(),
                        matmul.f32WideMaxNOverK(),
                        matmul.blasStorageMode(),
                        false,
                        matmul.blasThreads(),
                        matmul.openBlasArrayCopyThreads(),
                        matmul.openBlasNativeSegmentThreads()
                ),
                new config.runtime.Conv2dConfig(
                        conv2d.blasProvider(),
                        conv2d.f64BlasMinWork(),
                        conv2d.f32BlasMinWork(),
                        conv2d.f32RequireMgeK(),
                        conv2d.f32MaxNOverK(),
                        conv2d.bf16BlasMinWork(),
                        conv2d.bf16RequireMgeK(),
                        conv2d.bf16MaxNOverK()
                ),
                metadata.executionMode() == runtime.contract.ExecutionMode.FORWARD_BACKWARD
                        ? FusedExecutionPolicy.defaultsTraining()
                        : FusedExecutionPolicy.defaultsInference(),
                accelerator.toRuntimeConfig(),
                cpuStorageProfile,
                nativeCpuFailurePolicy,
                deviceTransferPolicy,
                config.runtime.NativeCpuMemoryConfig.disabled()
        );
    }

    /**
     * Returns the dtype for which this platform profile was calibrated.
     *
     * @return calibrated data type from profile metadata
     */
    public DataType dataType() {
        return metadata.dataType();
    }
}
