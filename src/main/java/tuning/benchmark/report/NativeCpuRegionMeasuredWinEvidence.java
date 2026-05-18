package tuning.benchmark.report;

import java.util.Map;

final class NativeCpuRegionMeasuredWinEvidence {
    static final double DEFAULT_REGION_WIN_RATIO = 0.95d;

    private NativeCpuRegionMeasuredWinEvidence() {
    }

    static boolean claimed(Map<String, Object> attrs) {
        Object value = attrs == null ? null : attrs.get("nativeCpuRegionMeasuredWin");
        return Boolean.TRUE.equals(value) || "true".equalsIgnoreCase(String.valueOf(value));
    }

    static boolean proven(Map<String, Object> attrs) {
        if (!claimed(attrs)) {
            return false;
        }
        double nativeMedianMs = doubleAttr(attrs.get("nativeCpuRegionNativeMedianMs"));
        double arrayMedianMs = doubleAttr(attrs.get("nativeCpuRegionArrayMedianMs"));
        double threshold = doubleAttr(attrs.get("nativeCpuRegionMeasuredWinThreshold"));
        if (!Double.isFinite(threshold) || threshold <= 0.0d) {
            threshold = DEFAULT_REGION_WIN_RATIO;
        }
        return Double.isFinite(nativeMedianMs)
                && Double.isFinite(arrayMedianMs)
                && nativeMedianMs > 0.0d
                && arrayMedianMs > 0.0d
                && nativeMedianMs <= arrayMedianMs * threshold;
    }

    static String describe(Map<String, Object> attrs) {
        return "{enabled=" + evidence(attrs, "nativeCpuRegionMeasuredWin")
                + ", nativeMedianMs=" + evidence(attrs, "nativeCpuRegionNativeMedianMs")
                + ", arrayMedianMs=" + evidence(attrs, "nativeCpuRegionArrayMedianMs")
                + ", threshold=" + evidence(attrs, "nativeCpuRegionMeasuredWinThreshold")
                + "}";
    }

    private static double doubleAttr(Object value) {
        if (value instanceof Number number) {
            return number.doubleValue();
        }
        if (value instanceof String text) {
            try {
                return Double.parseDouble(text);
            } catch (NumberFormatException ignored) {
                return Double.NaN;
            }
        }
        return Double.NaN;
    }

    private static String evidence(Map<String, Object> attrs, String key) {
        Object value = attrs == null ? null : attrs.get(key);
        if (value == null) {
            return "[]";
        }
        String text = String.valueOf(value);
        return text.isBlank() ? "[]" : text;
    }
}
