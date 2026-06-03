package backend.blas;

import java.util.OptionalInt;

/**
 * OpenBLAS runtime availability and symbol capability queries.
 */
public final class OpenBlasRuntime {
    private OpenBlasRuntime() {
    }

    public static boolean isAvailable() {
        return OpenBlasSymbols.get().available;
    }

    public static String unavailableReason() {
        return OpenBlasSymbols.get().reason;
    }

    public static boolean isBFloat16GemmAvailable() {
        return isBFloat16ToFloatGemmAvailable() || isBFloat16OutputGemmAvailable();
    }

    public static boolean isBFloat16ToFloatGemmAvailable() {
        OpenBlasSymbols symbols = OpenBlasSymbols.get();
        return symbols.available && symbols.sbgemm != null;
    }

    public static boolean isBFloat16OutputGemmAvailable() {
        OpenBlasSymbols symbols = OpenBlasSymbols.get();
        return symbols.available && symbols.bgemm != null;
    }

    public static boolean isFloat32GemmAvailable() {
        OpenBlasSymbols symbols = OpenBlasSymbols.get();
        return symbols.available && symbols.sgemm != null;
    }

    public static boolean isFloat64GemmAvailable() {
        OpenBlasSymbols symbols = OpenBlasSymbols.get();
        return symbols.available && symbols.dgemm != null;
    }

    public static String lookupSource() {
        OpenBlasSymbols symbols = OpenBlasSymbols.get();
        return symbols.source == null ? "UNAVAILABLE" : symbols.source.name();
    }

    public static String threadPolicy() {
        return "AUTO_UNCONTROLLED";
    }

    public static String threadPolicy(int requestedThreads) {
        if (requestedThreads <= 0) {
            return threadPolicy();
        }
        return "SET_NUM_THREADS(" + requestedThreads + ")";
    }

    public static OptionalInt getNumThreads() {
        OpenBlasSymbols symbols = OpenBlasSymbols.get();
        if (!symbols.available || symbols.getNumThreads == null) {
            return OptionalInt.empty();
        }
        try {
            return OptionalInt.of((int) symbols.getNumThreads.invokeExact());
        } catch (Throwable t) {
            throw new IllegalStateException("OpenBLAS openblas_get_num_threads call failed", t);
        }
    }

    public static boolean setNumThreads(int numThreads) {
        if (numThreads <= 0) {
            throw new IllegalArgumentException("numThreads must be positive: " + numThreads);
        }
        OpenBlasSymbols symbols = OpenBlasSymbols.get();
        if (!symbols.available || symbols.setNumThreads == null) {
            return false;
        }
        try {
            symbols.setNumThreads.invokeExact(numThreads);
            return true;
        } catch (Throwable t) {
            throw new IllegalStateException("OpenBLAS openblas_set_num_threads call failed", t);
        }
    }

    public static OptionalInt getParallelMode() {
        OpenBlasSymbols symbols = OpenBlasSymbols.get();
        if (!symbols.available || symbols.getParallel == null) {
            return OptionalInt.empty();
        }
        try {
            return OptionalInt.of((int) symbols.getParallel.invokeExact());
        } catch (Throwable t) {
            throw new IllegalStateException("OpenBLAS openblas_get_parallel call failed", t);
        }
    }

    public static String parallelModeDescription() {
        OptionalInt mode = getParallelMode();
        if (mode.isEmpty()) {
            return "UNAVAILABLE";
        }
        return switch (mode.getAsInt()) {
            case 0 -> "SEQUENTIAL";
            case 1 -> "PTHREADS";
            case 2 -> "OPENMP";
            default -> "UNKNOWN(" + mode.getAsInt() + ")";
        };
    }
}
