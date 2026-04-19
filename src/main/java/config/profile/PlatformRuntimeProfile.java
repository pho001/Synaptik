package config.profile;

import backend.blas.BlasProvider;
import config.backend.AttentionMatMulPolicy;
import config.backend.CpuMatMulMicroKernel;
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
        Conv2dPlatformProfile conv2d,
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
        conv2d = conv2d == null ? Conv2dPlatformProfile.fromMatmul(matmul) : conv2d;
        Objects.requireNonNull(fused, "fused cannot be null");
        Objects.requireNonNull(elementwiseDispatch, "elementwiseDispatch cannot be null");
        Objects.requireNonNull(reduction, "reduction cannot be null");
        Objects.requireNonNull(scheduler, "scheduler cannot be null");
        Objects.requireNonNull(materialization, "materialization cannot be null");
        Objects.requireNonNull(numerics, "numerics cannot be null");
    }

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
        this(metadata, matmul, Conv2dPlatformProfile.fromMatmul(matmul), fused, elementwiseDispatch, reduction, scheduler, materialization, numerics);
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
                        cpu.fusedCheapContiguousAsmVectorWidth(),
                        cpu.fusedCheapStridedAsmVectorWidth(),
                        cpu.fusedNonCheapContiguousAsmVectorWidth(),
                        cpu.fusedNonCheapStridedAsmVectorWidth()
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
                reduction.attentionVectorMinSize(),
                elementwiseDispatch.cheapParallelMinSize(),
                elementwiseDispatch.transcendentalParallelMinSize(),
                fused.fusedCheapParallelMinSize(),
                fused.fusedTranscendentalParallelMinSize(),
                reduction.reductionParallelMinSize(),
                reduction.attentionParallelMinSize(),
                materialization.contiguousMaterializeThreshold(),
                scheduler.lowCostTargetChunksPerWorker(),
                scheduler.mediumCostTargetChunksPerWorker(),
                scheduler.highCostTargetChunksPerWorker(),
                scheduler.minScalarChunkSize(),
                scheduler.minVectorChunkSize(),
                scheduler.minReductionChunkSize(),
                scheduler.commonPoolLowCostMaxWorkPerWorker(),
                fused.fusedCheapContiguousAsmVectorWidth(),
                fused.fusedCheapStridedAsmVectorWidth(),
                fused.fusedNonCheapContiguousAsmVectorWidth(),
                fused.fusedNonCheapStridedAsmVectorWidth(),
                reduction.sumAccuracyMode(),
                matmul.matMulParallelMinSize(),
                AttentionMatMulPolicy.AUTO,
                matmul.matMulMicroKernel() == null ? CpuMatMulMicroKernel.AUTO : matmul.matMulMicroKernel(),
                matmul.attentionMatMulMicroKernel() == null ? matmul.matMulMicroKernel() : matmul.attentionMatMulMicroKernel(),
                matmul.attentionMatMulTileM(),
                matmul.attentionMatMulTileN(),
                matmul.attentionMatMulTileK()
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
                metadata.executionMode() == backend.runtime.ExecutionMode.FORWARD_BACKWARD
                        ? FusedExecutionPolicy.defaultsTraining()
                        : FusedExecutionPolicy.defaultsInference()
        );
    }

    public DataType dataType() {
        return metadata.dataType();
    }
}
