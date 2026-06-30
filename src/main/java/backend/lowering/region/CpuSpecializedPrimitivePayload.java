package backend.lowering.region;

import planning.region.specialization.RegionSpecializationCandidate;
import planning.region.specialization.RegionSpecializationKind;

import java.util.Objects;

public record CpuSpecializedPrimitivePayload(
        RegionSpecializationCandidate candidate
) implements RegionBackendPayload {
    public CpuSpecializedPrimitivePayload {
        candidate = Objects.requireNonNull(candidate, "candidate cannot be null");
    }

    public RegionSpecializationKind kind() {
        return candidate.kind();
    }
}
