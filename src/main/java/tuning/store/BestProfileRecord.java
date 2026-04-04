package tuning.store;

import config.profile.ExecutionProfile;

import java.time.OffsetDateTime;

public record BestProfileRecord(
        HardwareFingerprint hardware,
        WorkloadFingerprint workload,
        ExecutionProfile profile,
        double score,
        OffsetDateTime updatedAt
) {
    public BestProfileRecord {
        if (hardware == null) {
            hardware = HardwareFingerprint.capture();
        }
        if (workload == null) {
            throw new IllegalArgumentException("workload cannot be null");
        }
        if (profile == null) {
            throw new IllegalArgumentException("profile cannot be null");
        }
        updatedAt = updatedAt == null ? OffsetDateTime.now() : updatedAt;
    }
}
