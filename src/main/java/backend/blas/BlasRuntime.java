package backend.blas;

/**
 * Runtime configuration accessors for optional CPU BLAS dispatch.
 *
 * <p>Invalid or missing system properties fall back to conservative defaults.
 * Selecting a provider does not guarantee native availability; callers must still
 * check the selected bridge before dispatch.</p>
 */
public final class BlasRuntime {
    /**
     * System property selecting the BLAS provider.
     */
    public static final String PROP_PROVIDER = "cg.cpu.blas.provider";
    /**
     * System property for the minimum matmul work estimate before BLAS dispatch.
     */
    public static final String PROP_MATMUL_MIN_WORK = "cg.cpu.blas.matmulMinWork";
    /**
     * System property enabling BLAS dispatch diagnostics.
     */
    public static final String PROP_DEBUG = "cg.cpu.blas.debug";
    /**
     * System property requiring {@code m >= k} for f32 BLAS dispatch.
     */
    public static final String PROP_F32_REQUIRE_M_GE_K = "cg.cpu.blas.f32RequireMgeK";
    /**
     * System property limiting {@code n / k} for f32 BLAS dispatch.
     */
    public static final String PROP_F32_MAX_N_OVER_K = "cg.cpu.blas.f32MaxNOverK";

    /**
     * Default provider keeps external BLAS disabled.
     */
    public static final String DEFAULT_PROVIDER = "NONE";
    /**
     * Default minimum matmul work estimate before BLAS dispatch.
     */
    public static final long DEFAULT_MATMUL_MIN_WORK = 2_000_000L;
    /**
     * Default diagnostics setting.
     */
    public static final boolean DEFAULT_DEBUG = false;
    /**
     * Default f32 shape gate requiring {@code m >= k}.
     */
    public static final boolean DEFAULT_F32_REQUIRE_M_GE_K = true;
    /**
     * Default f32 shape gate for {@code n / k}.
     */
    public static final double DEFAULT_F32_MAX_N_OVER_K = 3.0d;

    private BlasRuntime() {}

    /**
     * Returns the configured BLAS provider.
     */
    public static BlasProvider provider() {
        return BlasProvider.fromProperty(System.getProperty(PROP_PROVIDER, DEFAULT_PROVIDER));
    }

    /**
     * Returns the minimum matmul work estimate before BLAS dispatch is considered.
     */
    public static long matMulMinWork() {
        return parseLongProperty(PROP_MATMUL_MIN_WORK, DEFAULT_MATMUL_MIN_WORK);
    }

    /**
     * Returns whether BLAS dispatch diagnostics are enabled.
     */
    public static boolean debug() {
        return parseBooleanProperty(PROP_DEBUG, DEFAULT_DEBUG);
    }

    /**
     * Returns whether f32 BLAS dispatch requires {@code m >= k}.
     */
    public static boolean f32RequireMgeK() {
        return parseBooleanProperty(PROP_F32_REQUIRE_M_GE_K, DEFAULT_F32_REQUIRE_M_GE_K);
    }

    /**
     * Returns the maximum f32 {@code n / k} ratio allowed for BLAS dispatch.
     */
    public static double f32MaxNOverK() {
        return parseDoubleProperty(PROP_F32_MAX_N_OVER_K, DEFAULT_F32_MAX_N_OVER_K);
    }

    /**
     * Returns whether the OpenBLAS FFM provider is selected by configuration.
     */
    public static boolean isOpenBlasFfmEnabled() {
        return provider() == BlasProvider.OPENBLAS_FFM;
    }

    private static long parseLongProperty(String key, long fallback) {
        String raw = System.getProperty(key);
        if (raw == null || raw.isBlank()) {
            return fallback;
        }
        try {
            long parsed = Long.parseLong(raw.trim());
            return parsed > 0L ? parsed : fallback;
        } catch (NumberFormatException ignored) {
            return fallback;
        }
    }

    private static boolean parseBooleanProperty(String key, boolean fallback) {
        String raw = System.getProperty(key);
        if (raw == null || raw.isBlank()) {
            return fallback;
        }
        return Boolean.parseBoolean(raw.trim());
    }

    private static double parseDoubleProperty(String key, double fallback) {
        String raw = System.getProperty(key);
        if (raw == null || raw.isBlank()) {
            return fallback;
        }
        try {
            double parsed = Double.parseDouble(raw.trim());
            return parsed > 0.0d ? parsed : fallback;
        } catch (NumberFormatException ignored) {
            return fallback;
        }
    }
}
