package tuning.workload;

import java.util.Map;

/**
 * Metadata describing a workload instance for reporting and fingerprinting.
 *
 * @param name workload name
 * @param kind workload family
 * @param attributes additional fingerprint/report attributes
 */
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

    /**
     * Creates metadata without additional attributes.
     *
     * @param name workload name
     * @param kind workload family
     * @return workload metadata
     */
    public static WorkloadMetadata of(String name, WorkloadKind kind) {
        return new WorkloadMetadata(name, kind, Map.of());
    }
}
