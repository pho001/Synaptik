package backend.blas;

import java.util.Locale;

/**
 * CPU BLAS provider selected by runtime system properties.
 */
public enum BlasProvider {
    /**
     * Disable external BLAS dispatch and use the built-in CPU kernels.
     */
    NONE,
    /**
     * Use the OpenBLAS CBLAS bridge through the Java Foreign Function and Memory API.
     */
    OPENBLAS_FFM;

    /**
     * Parses a provider property value, returning {@link #NONE} for blank or unknown values.
     */
    public static BlasProvider fromProperty(String raw) {
        if (raw == null || raw.isBlank()) {
            return NONE;
        }
        try {
            return BlasProvider.valueOf(raw.trim().toUpperCase(Locale.ROOT));
        } catch (IllegalArgumentException ignored) {
            return NONE;
        }
    }
}
