package tuning.benchmark.report;

import java.util.Map;

final class NativeCpuPartitionMeasuredWinEvidence {
    static final double DEFAULT_PARTITION_WIN_RATIO = 0.95d;

    private NativeCpuPartitionMeasuredWinEvidence() {
    }

    static boolean claimed(Map<String, Object> attrs) {
        Object value = attrs == null ? null : attrs.get("nativeCpuPartitionMeasuredWin");
        return Boolean.TRUE.equals(value) || "true".equalsIgnoreCase(String.valueOf(value));
    }

    static boolean proven(Map<String, Object> attrs) {
        if (!claimed(attrs)) {
            return false;
        }
        double nativeMedianMs = doubleAttr(attrs.get("nativeCpuPartitionNativeMedianMs"));
        double arrayMedianMs = doubleAttr(attrs.get("nativeCpuPartitionArrayMedianMs"));
        double threshold = doubleAttr(attrs.get("nativeCpuPartitionMeasuredWinThreshold"));
        if (!Double.isFinite(threshold) || threshold <= 0.0d) {
            threshold = DEFAULT_PARTITION_WIN_RATIO;
        }
        return Double.isFinite(nativeMedianMs)
                && Double.isFinite(arrayMedianMs)
                && nativeMedianMs > 0.0d
                && arrayMedianMs > 0.0d
                && nativeMedianMs <= arrayMedianMs * threshold;
    }

    static String describe(Map<String, Object> attrs) {
        return "{enabled=" + evidence(attrs, "nativeCpuPartitionMeasuredWin")
                + ", nativeMedianMs=" + evidence(attrs, "nativeCpuPartitionNativeMedianMs")
                + ", arrayMedianMs=" + evidence(attrs, "nativeCpuPartitionArrayMedianMs")
                + ", threshold=" + evidence(attrs, "nativeCpuPartitionMeasuredWinThreshold")
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
