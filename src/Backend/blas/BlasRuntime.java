package Backend.blas;

public final class BlasRuntime {
    private static final BlasProvider PROVIDER = BlasProvider.fromProperty(
            System.getProperty("cg.cpu.blas.provider", "NONE")
    );
    private static final long MATMUL_MIN_WORK = parseLongProperty(
            "cg.cpu.blas.matmulMinWork",
            2_000_000L
    );
    private static final boolean DEBUG = Boolean.parseBoolean(
            System.getProperty("cg.cpu.blas.debug", "false")
    );

    private BlasRuntime() {}

    public static BlasProvider provider() {
        return PROVIDER;
    }

    public static long matMulMinWork() {
        return MATMUL_MIN_WORK;
    }

    public static boolean debug() {
        return DEBUG;
    }

    public static boolean isOpenBlasFfmEnabled() {
        return PROVIDER == BlasProvider.OPENBLAS_FFM;
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
}
