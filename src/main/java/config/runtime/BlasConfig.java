package config.runtime;

import backend.blas.BlasProvider;

import java.util.Objects;

/**
 * Runtime BLAS dispatch policy for matrix multiplication.
 *
 * <p>The config records the provider, minimum work threshold, shape filters for FLOAT32 dispatch, debug
 * flag, and requested thread counts. A thread value of {@code 0} means provider/default behavior.</p>
 *
 * @param provider BLAS provider; {@code null} becomes {@link BlasProvider#NONE}
 * @param matmulMinWork minimum estimated matmul work before BLAS is eligible
 * @param f32RequireMgeK whether regular FLOAT32 BLAS dispatch requires {@code M >= K}
 * @param f32MaxNOverK maximum {@code N / K} ratio for regular FLOAT32 BLAS dispatch
 * @param f32WideRequireMgeK whether wide FLOAT32 BLAS dispatch requires {@code M >= K}
 * @param f32WideMaxNOverK maximum {@code N / K} ratio for wide FLOAT32 BLAS dispatch
 * @param storageMode BLAS storage route policy
 * @param debug whether BLAS dispatch should emit debug diagnostics
 * @param threads fallback requested BLAS thread count
 * @param openBlasArrayCopyThreads requested OpenBLAS thread count for array-copy GEMM, or {@code 0}
 * @param openBlasNativeSegmentThreads requested OpenBLAS thread count for native-segment GEMM, or {@code 0}
 */
public record BlasConfig(
        BlasProvider provider,
        long matmulMinWork,
        boolean f32RequireMgeK,
        double f32MaxNOverK,
        boolean f32WideRequireMgeK,
        double f32WideMaxNOverK,
        BlasStorageMode storageMode,
        boolean debug,
        int threads,
        int openBlasArrayCopyThreads,
        int openBlasNativeSegmentThreads
) {
    public static final long DEFAULT_MATMUL_MIN_WORK = 2_000_000L;
    public static final boolean DEFAULT_F32_REQUIRE_M_GE_K = true;
    public static final double DEFAULT_F32_MAX_N_OVER_K = 3.0d;
    public static final int DEFAULT_THREADS = 0;

    public BlasConfig {
        provider = Objects.requireNonNullElse(provider, BlasProvider.NONE);
        storageMode = Objects.requireNonNullElse(storageMode, BlasStorageMode.CPU_ARRAY);
        matmulMinWork = matmulMinWork > 0 ? matmulMinWork : DEFAULT_MATMUL_MIN_WORK;
        f32MaxNOverK = f32MaxNOverK > 0.0d ? f32MaxNOverK : DEFAULT_F32_MAX_N_OVER_K;
        f32WideMaxNOverK = f32WideMaxNOverK > 0.0d ? f32WideMaxNOverK : f32MaxNOverK;
        threads = normalizeThreads(threads, "threads");
        openBlasArrayCopyThreads = normalizeThreads(openBlasArrayCopyThreads, "openBlasArrayCopyThreads");
        openBlasNativeSegmentThreads = normalizeThreads(openBlasNativeSegmentThreads, "openBlasNativeSegmentThreads");
    }

    public BlasConfig(
            BlasProvider provider,
            long matmulMinWork,
            boolean f32RequireMgeK,
            double f32MaxNOverK,
            boolean f32WideRequireMgeK,
            double f32WideMaxNOverK,
            BlasStorageMode storageMode,
            boolean debug,
            int threads
    ) {
        this(
                provider,
                matmulMinWork,
                f32RequireMgeK,
                f32MaxNOverK,
                f32WideRequireMgeK,
                f32WideMaxNOverK,
                storageMode,
                debug,
                threads,
                DEFAULT_THREADS,
                DEFAULT_THREADS
        );
    }

    public BlasConfig(
            BlasProvider provider,
            long matmulMinWork,
            boolean f32RequireMgeK,
            double f32MaxNOverK,
            boolean f32WideRequireMgeK,
            double f32WideMaxNOverK,
            boolean debug,
            int threads
    ) {
        this(provider, matmulMinWork, f32RequireMgeK, f32MaxNOverK, f32WideRequireMgeK, f32WideMaxNOverK, BlasStorageMode.CPU_ARRAY, debug, threads);
    }

    public BlasConfig(
            BlasProvider provider,
            long matmulMinWork,
            boolean f32RequireMgeK,
            double f32MaxNOverK,
            boolean f32WideRequireMgeK,
            double f32WideMaxNOverK,
            boolean debug
    ) {
        this(provider, matmulMinWork, f32RequireMgeK, f32MaxNOverK, f32WideRequireMgeK, f32WideMaxNOverK, BlasStorageMode.CPU_ARRAY, debug, DEFAULT_THREADS);
    }

    public BlasConfig(
            BlasProvider provider,
            long matmulMinWork,
            boolean f32RequireMgeK,
            double f32MaxNOverK,
            boolean f32WideRequireMgeK,
            double f32WideMaxNOverK,
            BlasStorageMode storageMode,
            boolean debug
    ) {
        this(provider, matmulMinWork, f32RequireMgeK, f32MaxNOverK, f32WideRequireMgeK, f32WideMaxNOverK, storageMode, debug, DEFAULT_THREADS);
    }

    public BlasConfig(
            BlasProvider provider,
            long matmulMinWork,
            boolean f32RequireMgeK,
            double f32MaxNOverK,
            boolean debug,
            int threads
    ) {
        this(provider, matmulMinWork, f32RequireMgeK, f32MaxNOverK, f32RequireMgeK, f32MaxNOverK, BlasStorageMode.CPU_ARRAY, debug, threads);
    }

    public BlasConfig(
            BlasProvider provider,
            long matmulMinWork,
            boolean f32RequireMgeK,
            double f32MaxNOverK,
            boolean debug
    ) {
        this(provider, matmulMinWork, f32RequireMgeK, f32MaxNOverK, f32RequireMgeK, f32MaxNOverK, BlasStorageMode.CPU_ARRAY, debug, DEFAULT_THREADS);
    }

    /**
     * Returns a config that disables BLAS dispatch while retaining default shape thresholds.
     *
     * @return BLAS-disabled runtime config
     */
    public static BlasConfig disabled() {
        return new BlasConfig(
                BlasProvider.NONE,
                DEFAULT_MATMUL_MIN_WORK,
                DEFAULT_F32_REQUIRE_M_GE_K,
                DEFAULT_F32_MAX_N_OVER_K,
                DEFAULT_F32_REQUIRE_M_GE_K,
                DEFAULT_F32_MAX_N_OVER_K,
                BlasStorageMode.CPU_ARRAY,
                false,
                DEFAULT_THREADS
        );
    }

    public BlasConfig withStorageMode(BlasStorageMode storageMode) {
        return new BlasConfig(
                provider,
                matmulMinWork,
                f32RequireMgeK,
                f32MaxNOverK,
                f32WideRequireMgeK,
                f32WideMaxNOverK,
                storageMode,
                debug,
                threads,
                openBlasArrayCopyThreads,
                openBlasNativeSegmentThreads
        );
    }

    public BlasConfig withThreads(int threads) {
        return new BlasConfig(
                provider,
                matmulMinWork,
                f32RequireMgeK,
                f32MaxNOverK,
                f32WideRequireMgeK,
                f32WideMaxNOverK,
                storageMode,
                debug,
                threads,
                openBlasArrayCopyThreads,
                openBlasNativeSegmentThreads
        );
    }

    public BlasConfig withOpenBlasRouteThreads(int arrayCopyThreads, int nativeSegmentThreads) {
        return new BlasConfig(
                provider,
                matmulMinWork,
                f32RequireMgeK,
                f32MaxNOverK,
                f32WideRequireMgeK,
                f32WideMaxNOverK,
                storageMode,
                debug,
                threads,
                arrayCopyThreads,
                nativeSegmentThreads
        );
    }

    public int openBlasArrayCopyEffectiveThreads() {
        return openBlasArrayCopyThreads > 0 ? openBlasArrayCopyThreads : threads;
    }

    public int openBlasNativeSegmentEffectiveThreads() {
        return openBlasNativeSegmentThreads > 0 ? openBlasNativeSegmentThreads : threads;
    }

    private static int normalizeThreads(int value, String name) {
        if (value < 0) {
            throw new IllegalArgumentException(name + " must be non-negative: " + value);
        }
        return value;
    }
}
