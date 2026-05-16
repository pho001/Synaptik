package backend.lowering.region;

import backend.cpu.fused.plan.FusedOperationPreparation;

import java.util.Objects;

public record CpuFusedRegionPayload(
        FusedOperationPreparation preparation
) implements RegionBackendPayload {
    public CpuFusedRegionPayload {
        preparation = Objects.requireNonNull(preparation, "preparation cannot be null");
    }
}
