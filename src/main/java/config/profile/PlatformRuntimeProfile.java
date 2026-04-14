package config.profile;

import backend.blas.BlasProvider;
import config.backend.AttentionMatMulPolicy;
import config.backend.CpuKernelConfig;
import config.backend.KernelTuningConfig;
import config.runtime.ApproximationConfig;
import config.runtime.BlasConfig;
import config.runtime.FusedExecutionPolicy;
import config.runtime.RuntimeConfig;
import tensor.DataType;

import java.time.OffsetDateTime;
import java.util.Objects;

public record PlatformRuntimeProfile(
        PlatformProfileMetadata metadata,
        MatmulPlatformProfile matmul,
        FusedPlatformProfile fused,
        ElementwiseDispatchPlatformProfile elementwiseDispatch,
        ReductionPlatformProfile reduction,
        SchedulerPlatformProfile scheduler,
        MaterializationPlatformProfile materialization,
        NumericsPlatformProfile numerics
) {
    public PlatformRuntimeProfile {
        Objects.requireNonNull(metadata, "metadata cannot be null");
        Objects.requireNonNull(matmul, "matmul cannot be null");
        Objects.requireNonNull(fused, "fused cannot be null");
        Objects.requireNonNull(elementwiseDispatch, "elementwiseDispatch cannot be null");
        Objects.requireNonNull(reduction, "reduction cannot be null");
        Objects.requireNonNull(scheduler, "scheduler cannot be null");
        Objects.requireNonNull(materialization, "materialization cannot be null");
        Objects.requireNonNull(numerics, "numerics cannot be null");
    }

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
                        profile.runtime().blas().f32RequireMgeK(),
                        profile.runtime().blas().f32MaxNOverK(),
                        cpu.loopUnrollFactor(),
                        cpu.matMulTileM(),
                        cpu.matMulTileN(),
                        cpu.matMulTileK(),
                        cpu.matMulParallelMinSize()
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
                        cpu.transcendentalVectorMinSize(),
                        cpu.cheapParallelMinSize(),
                        cpu.transcendentalParallelMinSize()
                ),
                new ReductionPlatformProfile(
                        cpu.reductionVectorMinSize(),
                        cpu.reductionParallelMinSize(),
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
                new MaterializationPlatformProfile(cpu.contiguousMaterializeThreshold()),
                new NumericsPlatformProfile(
                        profile.runtime().approximation().approxMode(),
                        profile.runtime().approximation().forceExactTranscendentals()
                )
        );
    }

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
                elementwiseDispatch.cheapParallelMinSize(),
                elementwiseDispatch.transcendentalParallelMinSize(),
                fused.fusedCheapParallelMinSize(),
                fused.fusedTranscendentalParallelMinSize(),
                reduction.reductionParallelMinSize(),
                materialization.contiguousMaterializeThreshold(),
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
                AttentionMatMulPolicy.AUTO
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
                        false,
                        matmul.blasThreads()
                ),
                metadata.executionMode() == backend.runtime.ExecutionMode.FORWARD_BACKWARD
                        ? FusedExecutionPolicy.defaultsTraining()
                        : FusedExecutionPolicy.defaultsInference()
        );
    }

    public DataType dataType() {
        return metadata.dataType();
    }
}
