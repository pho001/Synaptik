package config.runtime;

import backend.blas.BlasProvider;

import java.util.Objects;

public record BlasConfig(
        BlasProvider provider,
        long matmulMinWork,
        boolean f32RequireMgeK,
        double f32MaxNOverK,
        boolean debug,
        int threads
) {
    public static final long DEFAULT_MATMUL_MIN_WORK = 2_000_000L;
    public static final boolean DEFAULT_F32_REQUIRE_M_GE_K = true;
    public static final double DEFAULT_F32_MAX_N_OVER_K = 3.0d;
    public static final int DEFAULT_THREADS = 0;

    public BlasConfig {
        provider = Objects.requireNonNullElse(provider, BlasProvider.NONE);
        matmulMinWork = matmulMinWork > 0 ? matmulMinWork : DEFAULT_MATMUL_MIN_WORK;
        f32MaxNOverK = f32MaxNOverK > 0.0d ? f32MaxNOverK : DEFAULT_F32_MAX_N_OVER_K;
        threads = DEFAULT_THREADS;
    }

    public BlasConfig(
            BlasProvider provider,
            long matmulMinWork,
            boolean f32RequireMgeK,
            double f32MaxNOverK,
            boolean debug
    ) {
        this(provider, matmulMinWork, f32RequireMgeK, f32MaxNOverK, debug, DEFAULT_THREADS);
    }

    public static BlasConfig disabled() {
        return new BlasConfig(
                BlasProvider.NONE,
                DEFAULT_MATMUL_MIN_WORK,
                DEFAULT_F32_REQUIRE_M_GE_K,
                DEFAULT_F32_MAX_N_OVER_K,
                false,
                DEFAULT_THREADS
        );
    }
}
