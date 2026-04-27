package tuning.candidate;

import java.util.Map;
import java.util.TreeMap;

public final class CandidateIdentityFingerprint {
    private CandidateIdentityFingerprint() {
    }

    public static String of(Candidate candidate) {
        if (candidate == null) {
            throw new IllegalArgumentException("candidate cannot be null");
        }
        StringBuilder sb = new StringBuilder(512);
        sb.append("executable=").append(ExecutableProfileFingerprint.of(candidate)).append('|');
        sb.append("kind=").append(candidate.kind().name()).append('|');
        for (Map.Entry<String, String> entry : new TreeMap<>(candidate.metadata().toMap()).entrySet()) {
            sb.append("metadata.")
                    .append(entry.getKey())
                    .append('=')
                    .append(entry.getValue())
                    .append('|');
        }
        return ExecutableProfileFingerprint.sha256(sb.toString());
    }
}
