package config.profile;

import backend.blas.BlasProvider;

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
        int matMulParallelMinSize
) {
    public MatmulPlatformProfile {
        blasProvider = Objects.requireNonNull(blasProvider, "blasProvider cannot be null");
    }
}
