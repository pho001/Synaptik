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
        int attentionMatMulTileM,
        int attentionMatMulTileN,
        int attentionMatMulTileK,
        int matMulParallelMinSize,
        CpuMatMulMicroKernel matMulMicroKernel,
        CpuMatMulMicroKernel attentionMatMulMicroKernel
) {
    public MatmulPlatformProfile {
        blasProvider = Objects.requireNonNull(blasProvider, "blasProvider cannot be null");
        blasThreads = 0;
        attentionMatMulTileM = attentionMatMulTileM <= 0 ? matMulTileM : attentionMatMulTileM;
        attentionMatMulTileN = attentionMatMulTileN <= 0 ? matMulTileN : attentionMatMulTileN;
        attentionMatMulTileK = attentionMatMulTileK <= 0 ? matMulTileK : attentionMatMulTileK;
        matMulMicroKernel = matMulMicroKernel == null ? CpuMatMulMicroKernel.AUTO : matMulMicroKernel;
        attentionMatMulMicroKernel = attentionMatMulMicroKernel == null ? matMulMicroKernel : attentionMatMulMicroKernel;
    }
}
