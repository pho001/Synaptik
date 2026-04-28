package tuning.store;

import java.time.OffsetDateTime;
import java.util.LinkedHashMap;
import java.util.Map;
import tuning.candidate.CandidateKind;
import tuning.candidate.CandidateMetadata;

/**
 * Persisted per-candidate outcome from an autotune run.
 *
 * <p>History entries are append-only diagnostics. They record failures as well
 * as successful measurements so later search strategies and human reviewers can
 * understand why candidates were rejected.</p>
 *
 * @param fingerprint executable profile fingerprint
 * @param candidateName candidate name
 * @param valid whether validation and measurement succeeded
 * @param medianMs measured median latency, or positive infinity on failure
 * @param meanMs measured mean latency, or positive infinity on failure
 * @param score ranking score used by the run
 * @param failureReason validation or measurement failure reason
 * @param summary run-level summary string
 * @param timestamp entry creation time
 * @param hardware hardware fingerprint
 * @param workload workload fingerprint
 * @param candidateKind candidate category
 * @param candidateMetadata candidate provenance metadata
 * @param runtimeProfileId runtime profile id associated with the candidate
 * @param productionEligible whether the candidate may be promoted automatically
 */
public record TuningHistoryEntry(
        String fingerprint,
        String candidateName,
        boolean valid,
        double medianMs,
        double meanMs,
        double score,
        String failureReason,
        String summary,
        OffsetDateTime timestamp,
        HardwareFingerprint hardware,
        WorkloadFingerprint workload,
        CandidateKind candidateKind,
        CandidateMetadata candidateMetadata,
        String runtimeProfileId,
        boolean productionEligible
) {
    public TuningHistoryEntry {
        fingerprint = (fingerprint == null || fingerprint.isBlank()) ? candidateName : fingerprint;
        candidateName = (candidateName == null || candidateName.isBlank()) ? "candidate" : candidateName;
        failureReason = failureReason == null ? "" : failureReason;
        summary = summary == null ? "" : summary;
        timestamp = timestamp == null ? OffsetDateTime.now() : timestamp;
        if (hardware == null) {
            hardware = HardwareFingerprint.capture();
        }
        if (workload == null) {
            throw new IllegalArgumentException("workload fingerprint cannot be null");
        }
        candidateKind = candidateKind == null ? CandidateKind.GENERIC : candidateKind;
        candidateMetadata = candidateMetadata == null ? CandidateMetadata.generic() : candidateMetadata;
        runtimeProfileId = runtimeProfileId == null ? "" : runtimeProfileId;
    }

    public TuningHistoryEntry(
            String fingerprint,
            String candidateName,
            boolean valid,
            double medianMs,
            double meanMs,
            double score,
            String failureReason,
            String summary,
            OffsetDateTime timestamp,
            HardwareFingerprint hardware,
            WorkloadFingerprint workload
    ) {
        this(
                fingerprint,
                candidateName,
                valid,
                medianMs,
                meanMs,
                score,
                failureReason,
                summary,
                timestamp,
                hardware,
                workload,
                CandidateKind.GENERIC,
                CandidateMetadata.generic(),
                "",
                true
        );
    }

    /**
     * Converts this entry to a simple map for JSON-like stores.
     *
     * @return immutable map representation
     */
    public Map<String, Object> toMap() {
        Map<String, Object> map = new LinkedHashMap<>();
        map.put("fingerprint", fingerprint);
        map.put("candidateName", candidateName);
        map.put("valid", valid);
        map.put("medianMs", medianMs);
        map.put("meanMs", meanMs);
        map.put("score", score);
        map.put("failureReason", failureReason);
        map.put("summary", summary);
        map.put("timestamp", timestamp.toString());
        map.put("hardware", hardware.attributes());
        map.put("workload", workload.toMap());
        map.put("candidateKind", candidateKind.name());
        map.put("runtimeProfileId", runtimeProfileId);
        map.put("productionEligible", productionEligible);
        map.put("candidateMetadata", candidateMetadata.toMap());
        return Map.copyOf(map);
    }
}
