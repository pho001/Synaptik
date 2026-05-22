package backend.blas;

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
}
