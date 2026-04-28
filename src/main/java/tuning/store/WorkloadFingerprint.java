package tuning.store;

import config.profile.ExecutionProfile;
import tuning.workload.WorkloadMetadata;
import tuning.workload.WorkloadSpec;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.TreeMap;

/**
 * Normalized workload identity used to scope persisted tuning results.
 *
 * @param name workload name
 * @param kind workload kind
 * @param dataType execution dtype
 * @param mode execution mode
 * @param attributes sorted workload attributes
 */
public record WorkloadFingerprint(
        String name,
        String kind,
        String dataType,
        String mode,
        Map<String, Object> attributes
) {
    public WorkloadFingerprint {
        name = (name == null || name.isBlank()) ? "workload" : name;
        kind = (kind == null || kind.isBlank()) ? "GENERIC" : kind;
        dataType = (dataType == null || dataType.isBlank()) ? "UNKNOWN" : dataType;
        mode = (mode == null || mode.isBlank()) ? "FORWARD" : mode;
        attributes = attributes == null ? Map.of() : Map.copyOf(new TreeMap<>(attributes));
    }

    /**
     * Builds a workload fingerprint from workload metadata and profile context.
     *
     * @param workload workload specification, if available
     * @param metadata workload metadata, if available
     * @param profile execution profile, if available
     * @return workload fingerprint
     */
    public static WorkloadFingerprint of(WorkloadSpec workload, WorkloadMetadata metadata, ExecutionProfile profile) {
        return new WorkloadFingerprint(
                workload == null ? (metadata == null ? "workload" : metadata.name()) : workload.name(),
                workload == null ? (metadata == null ? "GENERIC" : metadata.kind().name()) : workload.kind().name(),
                profile == null ? "UNKNOWN" : profile.dataType().name(),
                profile == null ? "FORWARD" : profile.mode().name(),
                metadata == null ? Map.of() : metadata.attributes()
        );
    }

    /**
     * Parses a fingerprint key previously produced by {@link #key()}.
     *
     * @param key serialized key
     * @return parsed workload fingerprint
     */
    public static WorkloadFingerprint fromKey(String key) {
        if (key == null || key.isBlank()) {
            return new WorkloadFingerprint("workload", "GENERIC", "UNKNOWN", "FORWARD", Map.of());
        }
        java.util.LinkedHashMap<String, Object> attrs = new java.util.LinkedHashMap<>();
        String name = "workload";
        String kind = "GENERIC";
        String dataType = "UNKNOWN";
        String mode = "FORWARD";
        for (String token : key.split("\\|")) {
            int idx = token.indexOf('=');
            if (idx <= 0) {
                continue;
            }
            String k = token.substring(0, idx);
            String v = token.substring(idx + 1);
            switch (k) {
                case "name" -> name = v;
                case "kind" -> kind = v;
                case "dtype" -> dataType = v;
                case "mode" -> mode = v;
                default -> attrs.put(k, v);
            }
        }
        return new WorkloadFingerprint(name, kind, dataType, mode, attrs);
    }

    /**
     * @return stable key suitable for persistence lookups
     */
    public String key() {
        StringBuilder sb = new StringBuilder();
        sb.append("name=").append(name)
                .append("|kind=").append(kind)
                .append("|dtype=").append(dataType)
                .append("|mode=").append(mode);
        for (Map.Entry<String, Object> entry : attributes.entrySet()) {
            sb.append('|').append(entry.getKey()).append('=').append(entry.getValue());
        }
        return sb.toString();
    }

    /**
     * @return map representation for JSON-like stores
     */
    public Map<String, Object> toMap() {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("name", name);
        map.put("kind", kind);
        map.put("dataType", dataType);
        map.put("mode", mode);
        map.put("attributes", attributes);
        return Map.copyOf(map);
    }
}
