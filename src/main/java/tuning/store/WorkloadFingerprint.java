package tuning.store;

import config.profile.ExecutionProfile;
import tuning.workload.WorkloadMetadata;
import tuning.workload.WorkloadSpec;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.TreeMap;

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

    public static WorkloadFingerprint of(WorkloadSpec workload, WorkloadMetadata metadata, ExecutionProfile profile) {
        return new WorkloadFingerprint(
                workload == null ? (metadata == null ? "workload" : metadata.name()) : workload.name(),
                workload == null ? (metadata == null ? "GENERIC" : metadata.kind().name()) : workload.kind().name(),
                profile == null ? "UNKNOWN" : profile.dataType().name(),
                profile == null ? "FORWARD" : profile.mode().name(),
                metadata == null ? Map.of() : metadata.attributes()
        );
    }

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
