package tuning.store;

import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;

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

    public static HardwareFingerprint capture() {
        return new HardwareFingerprint(
                System.getProperty("os.name", "unknown"),
                System.getProperty("os.arch", "unknown"),
                System.getProperty("java.vm.name", "unknown"),
                System.getProperty("java.vendor", "unknown"),
                Runtime.getRuntime().availableProcessors()
        );
    }

    public String key() {
        return "os=" + os + "|arch=" + arch + "|vm=" + vm + "|vendor=" + vendor + "|cores=" + cores;
    }

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
}
