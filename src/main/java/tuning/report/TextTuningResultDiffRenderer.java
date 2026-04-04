package tuning.report;

import java.util.Locale;

public final class TextTuningResultDiffRenderer {
    private TextTuningResultDiffRenderer() {
    }

    public static String render(TuningResultDiff diff) {
        if (diff == null) {
            throw new IllegalArgumentException("diff cannot be null");
        }
        StringBuilder sb = new StringBuilder();
        sb.append("Tuning Result Diff\n");
        sb.append("previousBestProfile=").append(blankToNa(diff.previousBestProfile())).append('\n');
        sb.append("currentBestProfile=").append(blankToNa(diff.currentBestProfile())).append('\n');
        sb.append("previousBestMedianMs=").append(formatDouble(diff.previousBestMedianMs())).append('\n');
        sb.append("currentBestMedianMs=").append(formatDouble(diff.currentBestMedianMs())).append('\n');
        sb.append("bestSpeedupVsPrevious=").append(formatRatio(diff.bestSpeedupVsPrevious())).append('\n');
        sb.append("previousFinalistCount=").append(diff.previousFinalistCount()).append('\n');
        sb.append("currentFinalistCount=").append(diff.currentFinalistCount()).append('\n');
        return sb.toString();
    }

    private static String formatDouble(double value) {
        return Double.isFinite(value) ? String.format(Locale.US, "%.6f", value) : "n/a";
    }

    private static String formatRatio(double value) {
        return Double.isFinite(value) ? String.format(Locale.US, "%.3fx", value) : "n/a";
    }

    private static String blankToNa(String value) {
        return value == null || value.isBlank() ? "n/a" : value;
    }
}
