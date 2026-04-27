package tuning.store;

import config.profile.ExecutionProfile;
import tuning.candidate.CandidateKind;
import tuning.candidate.CandidateMetadata;

import java.time.OffsetDateTime;

public record BestProfileRecord(
        HardwareFingerprint hardware,
        WorkloadFingerprint workload,
        ExecutionProfile profile,
        double score,
        OffsetDateTime updatedAt,
        String autotuneKind,
        String graphAutotuneMode,
        CandidateKind candidateKind,
        CandidateMetadata candidateMetadata,
        String runtimeProfileId,
        boolean productionEligible
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
        autotuneKind = autotuneKind == null || autotuneKind.isBlank() ? "unknown" : autotuneKind;
        graphAutotuneMode = graphAutotuneMode == null ? "" : graphAutotuneMode;
        candidateKind = candidateKind == null ? CandidateKind.GENERIC : candidateKind;
        candidateMetadata = candidateMetadata == null ? CandidateMetadata.generic() : candidateMetadata;
        runtimeProfileId = runtimeProfileId == null ? "" : runtimeProfileId;
    }

    public BestProfileRecord(
            HardwareFingerprint hardware,
            WorkloadFingerprint workload,
            ExecutionProfile profile,
            double score,
            OffsetDateTime updatedAt
    ) {
        this(
                hardware,
                workload,
                profile,
                score,
                updatedAt,
                "legacy",
                "",
                CandidateKind.GENERIC,
                CandidateMetadata.generic(),
                "",
                true
        );
    }
}
