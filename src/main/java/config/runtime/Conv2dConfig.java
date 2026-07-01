package config.runtime;

import tensor.DataType;

import java.util.Objects;

/**
 * Legacy runtime dispatch policy for conv2d lowering.
 *
 * <p>Conv2d used dtype-specific thresholds because F64, F32, and BF16 have different crossover points
 * and BLAS constraints. Unsupported non-floating dtypes return conservative values from helper methods.</p>
 *
 * @param provider BLAS provider for historical conv2d dispatch
 * @param f64MinWork minimum F64 work before BLAS is eligible
 * @param f32MinWork minimum F32 work before BLAS is eligible
 * @param f32RequireMgeK whether F32 dispatch requires {@code M >= K}
 * @param f32MaxNOverK maximum {@code N / K} ratio for F32 dispatch
 * @param bf16MinWork minimum BF16 work before BLAS is eligible
 * @param bf16RequireMgeK whether BF16 dispatch requires {@code M >= K}
 * @param bf16MaxNOverK maximum {@code N / K} ratio for BF16 dispatch
 */
public record Conv2dConfig(
        BlasProvider provider,
        long f64MinWork,
        long f32MinWork,
        boolean f32RequireMgeK,
        double f32MaxNOverK,
        long bf16MinWork,
        boolean bf16RequireMgeK,
        double bf16MaxNOverK
) {
    public static final long DEFAULT_F64_MIN_WORK = BlasConfig.DEFAULT_MATMUL_MIN_WORK;
    public static final long DEFAULT_F32_MIN_WORK = BlasConfig.DEFAULT_MATMUL_MIN_WORK;
    public static final long DEFAULT_BF16_MIN_WORK = BlasConfig.DEFAULT_MATMUL_MIN_WORK;
    public static final boolean DEFAULT_F32_REQUIRE_M_GE_K = BlasConfig.DEFAULT_F32_REQUIRE_M_GE_K;
    public static final boolean DEFAULT_BF16_REQUIRE_M_GE_K = BlasConfig.DEFAULT_F32_REQUIRE_M_GE_K;
    public static final double DEFAULT_F32_MAX_N_OVER_K = BlasConfig.DEFAULT_F32_MAX_N_OVER_K;
    public static final double DEFAULT_BF16_MAX_N_OVER_K = BlasConfig.DEFAULT_F32_MAX_N_OVER_K;

    public Conv2dConfig {
        provider = Objects.requireNonNullElse(provider, BlasProvider.NONE);
        f64MinWork = f64MinWork > 0 ? f64MinWork : DEFAULT_F64_MIN_WORK;
        f32MinWork = f32MinWork > 0 ? f32MinWork : DEFAULT_F32_MIN_WORK;
        f32MaxNOverK = f32MaxNOverK > 0.0d ? f32MaxNOverK : DEFAULT_F32_MAX_N_OVER_K;
        bf16MinWork = bf16MinWork > 0 ? bf16MinWork : DEFAULT_BF16_MIN_WORK;
        bf16MaxNOverK = bf16MaxNOverK > 0.0d ? bf16MaxNOverK : DEFAULT_BF16_MAX_N_OVER_K;
    }

    /**
     * Returns a conv2d config with BLAS provider disabled.
     *
     * @return BLAS-disabled conv2d config
     */
    public static Conv2dConfig disabled() {
        return new Conv2dConfig(
                BlasProvider.NONE,
                DEFAULT_F64_MIN_WORK,
                DEFAULT_F32_MIN_WORK,
                DEFAULT_F32_REQUIRE_M_GE_K,
                DEFAULT_F32_MAX_N_OVER_K,
                DEFAULT_BF16_MIN_WORK,
                DEFAULT_BF16_REQUIRE_M_GE_K,
                DEFAULT_BF16_MAX_N_OVER_K
        );
    }

    /**
     * Seeds conv2d dispatch thresholds from the general matmul BLAS config.
     *
     * @param blas BLAS config; {@code null} disables conv2d BLAS dispatch
     * @return conv2d config using the BLAS provider and thresholds as fallback values
     */
    public static Conv2dConfig fromBlasConfig(BlasConfig blas) {
        if (blas == null) {
            return disabled();
        }
        return new Conv2dConfig(
                blas.provider(),
                blas.matmulMinWork(),
                blas.matmulMinWork(),
                blas.f32RequireMgeK(),
                blas.f32MaxNOverK(),
                blas.matmulMinWork(),
                blas.f32RequireMgeK(),
                blas.f32MaxNOverK()
        );
    }

    /**
     * Returns the dtype-specific minimum work threshold.
     *
     * @param dataType dtype to inspect
     * @return configured threshold, or {@link Long#MAX_VALUE} for unsupported dtypes
     */
    public long minWork(DataType dataType) {
        return switch (dataType) {
            case FLOAT64 -> f64MinWork;
            case FLOAT32 -> f32MinWork;
            case BFLOAT16 -> bf16MinWork;
            default -> Long.MAX_VALUE;
        };
    }

    /**
     * Returns whether the dtype-specific dispatch requires {@code M >= K}.
     *
     * @param dataType dtype to inspect
     * @return dtype-specific shape guard, or {@code false} for dtypes without such guard
     */
    public boolean requireMgeK(DataType dataType) {
        return switch (dataType) {
            case FLOAT32 -> f32RequireMgeK;
            case BFLOAT16 -> bf16RequireMgeK;
            default -> false;
        };
    }

    /**
     * Returns the dtype-specific maximum {@code N / K} ratio.
     *
     * @param dataType dtype to inspect
     * @return configured ratio, or positive infinity for dtypes without a ratio guard
     */
    public double maxNOverK(DataType dataType) {
        return switch (dataType) {
            case FLOAT32 -> f32MaxNOverK;
            case BFLOAT16 -> bf16MaxNOverK;
            default -> Double.POSITIVE_INFINITY;
        };
    }
}
