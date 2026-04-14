package config.profile;

import backend.blas.BlasProvider;
import config.backend.CpuMatMulMicroKernel;

import java.util.Objects;

public record MatmulPlatformProfile(
        BlasProvider blasProvider,
        long blasMatmulMinWork,
        int blasThreads,
        boolean f32RequireMgeK,
        double f32MaxNOverK,
        int loopUnrollFactor,
        int matMulTileM,
        int matMulTileN,
        int matMulTileK,
        int matMulParallelMinSize,
        CpuMatMulMicroKernel matMulMicroKernel
) {
    public MatmulPlatformProfile {
        blasProvider = Objects.requireNonNull(blasProvider, "blasProvider cannot be null");
        matMulMicroKernel = matMulMicroKernel == null ? CpuMatMulMicroKernel.AUTO : matMulMicroKernel;
    }
}
