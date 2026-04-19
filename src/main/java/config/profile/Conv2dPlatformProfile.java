package config.profile;

import backend.blas.BlasProvider;

import java.util.Objects;

public record Conv2dPlatformProfile(
        BlasProvider blasProvider,
        long f64BlasMinWork,
        long f32BlasMinWork,
        boolean f32RequireMgeK,
        double f32MaxNOverK,
        long bf16BlasMinWork,
        boolean bf16RequireMgeK,
        double bf16MaxNOverK
) {
    public Conv2dPlatformProfile {
        blasProvider = Objects.requireNonNullElse(blasProvider, BlasProvider.NONE);
        f64BlasMinWork = Math.max(1L, f64BlasMinWork);
        f32BlasMinWork = Math.max(1L, f32BlasMinWork);
        f32MaxNOverK = f32MaxNOverK > 0.0d ? f32MaxNOverK : 3.0d;
        bf16BlasMinWork = Math.max(1L, bf16BlasMinWork);
        bf16MaxNOverK = bf16MaxNOverK > 0.0d ? bf16MaxNOverK : 3.0d;
    }

    public static Conv2dPlatformProfile fromMatmul(MatmulPlatformProfile matmul) {
        Objects.requireNonNull(matmul, "matmul cannot be null");
        return new Conv2dPlatformProfile(
                matmul.blasProvider(),
                matmul.blasMatmulMinWork(),
                matmul.blasMatmulMinWork(),
                matmul.f32RequireMgeK(),
                matmul.f32MaxNOverK(),
                matmul.blasMatmulMinWork(),
                matmul.f32RequireMgeK(),
                matmul.f32MaxNOverK()
        );
    }
}
