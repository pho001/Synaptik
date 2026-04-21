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
        boolean f32WideRequireMgeK,
        double f32WideMaxNOverK,
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
        f32WideMaxNOverK = f32WideMaxNOverK > 0.0d ? f32WideMaxNOverK : f32MaxNOverK;
    }

    public MatmulPlatformProfile(
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
        this(
                blasProvider,
                blasMatmulMinWork,
                blasThreads,
                f32RequireMgeK,
                f32MaxNOverK,
                f32RequireMgeK,
                f32MaxNOverK,
                loopUnrollFactor,
                matMulTileM,
                matMulTileN,
                matMulTileK,
                attentionMatMulTileM,
                attentionMatMulTileN,
                attentionMatMulTileK,
                matMulParallelMinSize,
                matMulMicroKernel,
                attentionMatMulMicroKernel
        );
    }
}
