package tuning.store;

import java.time.OffsetDateTime;
import java.util.LinkedHashMap;
import java.util.Map;
import tuning.candidate.CandidateKind;
import tuning.candidate.CandidateMetadata;

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
