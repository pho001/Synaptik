package config.profile;

import config.runtime.BlasProvider;

import java.util.Objects;

/**
 * Legacy calibrated conv2d dispatch thresholds for one platform/dtype family.
 *
 * <p>This profile records the provider and dtype-specific thresholds used by historical conv2d
 * runtime dispatch. Invalid
 * non-positive thresholds are normalized to conservative positive values.</p>
 *
 * @param blasProvider BLAS provider selected for historical conv2d dispatch
 * @param f64BlasMinWork minimum F64 conv2d work before BLAS is eligible
 * @param f32BlasMinWork minimum F32 conv2d work before BLAS is eligible
 * @param f32RequireMgeK whether F32 BLAS dispatch requires {@code M >= K}
 * @param f32MaxNOverK maximum {@code N / K} ratio for regular F32 BLAS dispatch
 * @param bf16BlasMinWork minimum BF16 conv2d work before BLAS is eligible
 * @param bf16RequireMgeK whether BF16 BLAS dispatch requires {@code M >= K}
 * @param bf16MaxNOverK maximum {@code N / K} ratio for BF16 BLAS dispatch
 */
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

    /**
     * Builds a conservative conv2d dispatch profile from matmul calibration settings.
     *
     * <p>This is used as a fallback seed when an older profile or runtime path has no explicit conv2d
     * section. Calibration should eventually replace it with conv2d-specific measured values.</p>
     *
     * @param matmul matmul profile to reuse as seed; must not be {@code null}
     * @return conv2d profile seeded from matmul provider and thresholds
     */
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
