package tuning.autotune.report;

import java.util.Locale;

public final class JsonTuningResultDiffRenderer {
    private JsonTuningResultDiffRenderer() {
    }

    public static String render(TuningResultDiff diff) {
        if (diff == null) {
            throw new IllegalArgumentException("diff cannot be null");
        }
        return "{\n"
                + "  \"previousBestProfile\": \"" + escape(diff.previousBestProfile()) + "\",\n"
                + "  \"currentBestProfile\": \"" + escape(diff.currentBestProfile()) + "\",\n"
                + "  \"previousBestMedianMs\": " + format(diff.previousBestMedianMs()) + ",\n"
                + "  \"currentBestMedianMs\": " + format(diff.currentBestMedianMs()) + ",\n"
                + "  \"bestSpeedupVsPrevious\": " + format(diff.bestSpeedupVsPrevious()) + ",\n"
                + "  \"previousFinalistCount\": " + diff.previousFinalistCount() + ",\n"
                + "  \"currentFinalistCount\": " + diff.currentFinalistCount() + "\n"
                + "}\n";
    }

    private static String format(double value) {
        return Double.isFinite(value) ? String.format(Locale.US, "%.6f", value) : "null";
    }

    private static String escape(String value) {
        return value == null ? "" : value.replace("\\", "\\\\").replace("\"", "\\\"");
    }
}
