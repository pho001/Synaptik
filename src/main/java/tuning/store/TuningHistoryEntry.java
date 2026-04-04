package tuning.store;

import java.time.OffsetDateTime;
import java.util.LinkedHashMap;
import java.util.Map;

public record TuningHistoryEntry(
        String candidateName,
        boolean valid,
        double medianMs,
        double meanMs,
        double score,
        String summary,
        OffsetDateTime timestamp,
        HardwareFingerprint hardware,
        WorkloadFingerprint workload
) {
    public TuningHistoryEntry {
        candidateName = (candidateName == null || candidateName.isBlank()) ? "candidate" : candidateName;
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
        map.put("candidateName", candidateName);
        map.put("valid", valid);
        map.put("medianMs", medianMs);
        map.put("meanMs", meanMs);
        map.put("score", score);
        map.put("summary", summary);
        map.put("timestamp", timestamp.toString());
        map.put("hardware", hardware.attributes());
        map.put("workload", workload.toMap());
        return Map.copyOf(map);
    }
}
