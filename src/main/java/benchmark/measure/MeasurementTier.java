package benchmark.measure;

import java.util.Locale;

public enum MeasurementTier {
    SCOUT,
    PRESCREEN,
    REFINE,
    FINAL,
    OTHER;

    public static MeasurementTier fromName(String raw) {
        if (raw == null || raw.isBlank()) {
            return OTHER;
        }
        try {
            return MeasurementTier.valueOf(raw.trim().toUpperCase(Locale.ROOT));
        } catch (IllegalArgumentException ex) {
            return OTHER;
        }
    }
}
