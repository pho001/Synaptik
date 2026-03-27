package Backend.blas;

public final class BlasRuntime {
    public static final String PROP_PROVIDER = "cg.cpu.blas.provider";
    public static final String PROP_MATMUL_MIN_WORK = "cg.cpu.blas.matmulMinWork";
    public static final String PROP_DEBUG = "cg.cpu.blas.debug";
    public static final String PROP_F32_REQUIRE_M_GE_K = "cg.cpu.blas.f32RequireMgeK";
    public static final String PROP_F32_MAX_N_OVER_K = "cg.cpu.blas.f32MaxNOverK";

    public static final String DEFAULT_PROVIDER = "NONE";
    public static final long DEFAULT_MATMUL_MIN_WORK = 2_000_000L;
    public static final boolean DEFAULT_DEBUG = false;
    public static final boolean DEFAULT_F32_REQUIRE_M_GE_K = true;
    public static final double DEFAULT_F32_MAX_N_OVER_K = 3.0d;

    private BlasRuntime() {}

    public static BlasProvider provider() {
        return BlasProvider.fromProperty(System.getProperty(PROP_PROVIDER, DEFAULT_PROVIDER));
    }

    public static long matMulMinWork() {
        return parseLongProperty(PROP_MATMUL_MIN_WORK, DEFAULT_MATMUL_MIN_WORK);
    }

    public static boolean debug() {
        return parseBooleanProperty(PROP_DEBUG, DEFAULT_DEBUG);
    }

    public static boolean f32RequireMgeK() {
        return parseBooleanProperty(PROP_F32_REQUIRE_M_GE_K, DEFAULT_F32_REQUIRE_M_GE_K);
    }

    public static double f32MaxNOverK() {
        return parseDoubleProperty(PROP_F32_MAX_N_OVER_K, DEFAULT_F32_MAX_N_OVER_K);
    }

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
