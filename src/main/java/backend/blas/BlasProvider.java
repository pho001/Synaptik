package backend.blas;

import java.util.Locale;

public enum BlasProvider {
    NONE,
    OPENBLAS_FFM;

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
