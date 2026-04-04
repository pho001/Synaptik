package tuning.workload;

import java.util.Map;

public record WorkloadMetadata(
        String name,
        WorkloadKind kind,
        Map<String, Object> attributes
) {
    public WorkloadMetadata {
        name = (name == null || name.isBlank()) ? "workload" : name;
        kind = kind == null ? WorkloadKind.GENERIC : kind;
        attributes = attributes == null ? Map.of() : Map.copyOf(attributes);
    }

    public static WorkloadMetadata of(String name, WorkloadKind kind) {
        return new WorkloadMetadata(name, kind, Map.of());
    }
}
