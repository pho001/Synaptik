package backend.lowering.partition;

import java.util.Objects;

public record CpuFusedPartitionPayload(
        Object preparation
) implements PartitionBackendPayload {
    public CpuFusedPartitionPayload {
        preparation = Objects.requireNonNull(preparation, "preparation cannot be null");
    }

    public <T> T requirePreparation(Class<T> preparationType) {
        Objects.requireNonNull(preparationType, "preparationType cannot be null");
        if (!preparationType.isInstance(preparation)) {
            throw new IllegalStateException("CPU fused partition payload requires " + preparationType.getSimpleName());
        }
        return preparationType.cast(preparation);
    }
}
