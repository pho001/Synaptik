package backend.lowering.region;

import java.util.Objects;

public record CpuFusedRegionPayload(
        Object preparation
) implements RegionBackendPayload {
    public CpuFusedRegionPayload {
        preparation = Objects.requireNonNull(preparation, "preparation cannot be null");
    }

    public <T> T requirePreparation(Class<T> preparationType) {
        Objects.requireNonNull(preparationType, "preparationType cannot be null");
        if (!preparationType.isInstance(preparation)) {
            throw new IllegalStateException("CPU fused region payload requires " + preparationType.getSimpleName());
        }
        return preparationType.cast(preparation);
    }
}
