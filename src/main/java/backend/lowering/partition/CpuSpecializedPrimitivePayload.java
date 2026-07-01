package backend.lowering.partition;

import planning.partition.specialization.PartitionSpecializationCandidate;
import planning.partition.specialization.PartitionSpecializationKind;

import java.util.Objects;

public record CpuSpecializedPrimitivePayload(
        PartitionSpecializationCandidate candidate
) implements PartitionBackendPayload {
    public CpuSpecializedPrimitivePayload {
        candidate = Objects.requireNonNull(candidate, "candidate cannot be null");
    }

    public PartitionSpecializationKind kind() {
        return candidate.kind();
    }
}
