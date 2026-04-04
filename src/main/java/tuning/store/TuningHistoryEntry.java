package tuning.store;

import java.time.OffsetDateTime;
import java.util.LinkedHashMap;
import java.util.Map;

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
        WorkloadFingerprint workload
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
        return Map.copyOf(map);
    }
}
