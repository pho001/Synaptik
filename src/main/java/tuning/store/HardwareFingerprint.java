package tuning.store;

import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;

/**
 * Normalized hardware identity used to scope persisted tuning results.
 *
 * @param os normalized operating system name
 * @param arch normalized CPU architecture
 * @param vm normalized JVM name
 * @param vendor normalized JVM vendor
 * @param cores available processor count
 */
public record HardwareFingerprint(
        String os,
        String arch,
        String vm,
        String vendor,
        int cores
) {
    public HardwareFingerprint {
        os = normalize(os);
        arch = normalize(arch);
        vm = normalize(vm);
        vendor = normalize(vendor);
        cores = Math.max(1, cores);
    }

    /**
     * Captures the current JVM process hardware/runtime fingerprint.
     *
     * @return current hardware fingerprint
     */
    public static HardwareFingerprint capture() {
        return new HardwareFingerprint(
                System.getProperty("os.name", "unknown"),
                System.getProperty("os.arch", "unknown"),
                System.getProperty("java.vm.name", "unknown"),
                System.getProperty("java.vendor", "unknown"),
                Runtime.getRuntime().availableProcessors()
        );
    }

    /**
     * Parses a fingerprint key previously produced by {@link #key()}.
     *
     * @param key serialized key
     * @return parsed fingerprint, or current capture for blank keys
     */
    public static HardwareFingerprint fromKey(String key) {
        if (key == null || key.isBlank()) {
            return capture();
        }
        java.util.Map<String, String> values = splitKey(key);
        return new HardwareFingerprint(
                values.getOrDefault("os", "unknown"),
                values.getOrDefault("arch", "unknown"),
                values.getOrDefault("vm", "unknown"),
                values.getOrDefault("vendor", "unknown"),
                parseInt(values.get("cores"), Runtime.getRuntime().availableProcessors())
        );
    }

    /**
     * @return stable key suitable for simple persistence lookups
     */
    public String key() {
        return "os=" + os + "|arch=" + arch + "|vm=" + vm + "|vendor=" + vendor + "|cores=" + cores;
    }

    /**
     * @return map representation for reports and JSON stores
     */
    public Map<String, Object> attributes() {
        Map<String, Object> attrs = new LinkedHashMap<>();
        attrs.put("os", os);
        attrs.put("arch", arch);
        attrs.put("vm", vm);
        attrs.put("vendor", vendor);
        attrs.put("cores", cores);
        return Map.copyOf(attrs);
    }

    private static String normalize(String value) {
        if (value == null || value.isBlank()) {
            return "unknown";
        }
        return value.trim()
                .toLowerCase(Locale.ROOT)
                .replace(' ', '_')
                .replace('\t', '_');
    }

    private static java.util.Map<String, String> splitKey(String key) {
        java.util.LinkedHashMap<String, String> out = new java.util.LinkedHashMap<>();
        for (String token : key.split("\\|")) {
            int idx = token.indexOf('=');
            if (idx <= 0) {
                continue;
            }
            out.put(token.substring(0, idx), token.substring(idx + 1));
        }
        return out;
    }

    private static int parseInt(String value, int fallback) {
        try {
            return value == null ? fallback : Integer.parseInt(value);
        } catch (Exception ignored) {
            return fallback;
        }
    }
}
