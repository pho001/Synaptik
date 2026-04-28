package tuning.store;

import config.profile.ExecutionProfile;
import tuning.candidate.CandidateKind;
import tuning.candidate.CandidateMetadata;

import java.time.OffsetDateTime;

/**
 * Persisted best-profile selection from autotune.
 *
 * <p>The record combines hardware/workload fingerprints with candidate metadata
 * so consumers can decide whether a saved profile is applicable and safe to
 * promote.</p>
 *
 * @param hardware hardware fingerprint captured at selection time
 * @param workload workload fingerprint used for applicability checks
 * @param profile selected execution profile
 * @param score primary latency score, usually median milliseconds
 * @param updatedAt save time
 * @param autotuneKind source autotune flow, such as graph or legacy
 * @param graphAutotuneMode graph autotune mode, when applicable
 * @param candidateKind selected candidate category
 * @param candidateMetadata selected candidate provenance metadata
 * @param runtimeProfileId runtime profile id associated with the selection
 * @param productionEligible whether automated promotion is allowed
 */
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
