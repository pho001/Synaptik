package Benchmark.autotune;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.OffsetDateTime;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public final class UnsafeCandidateHistory {
    private final String contextSignature;
    private final Map<String, UnsafeCandidateRecord> unsafeByFingerprint;
    private boolean dirty;

    private UnsafeCandidateHistory(String contextSignature, Map<String, UnsafeCandidateRecord> unsafeByFingerprint) {
        this.contextSignature = contextSignature;
        this.unsafeByFingerprint = unsafeByFingerprint;
    }

    public static UnsafeCandidateHistory empty(String contextSignature) {
        return new UnsafeCandidateHistory(contextSignature, new HashMap<>());
    }

    public static UnsafeCandidateHistory load(Path path, String contextSignature) {
        if (!Files.exists(path)) {
            return empty(contextSignature);
        }
        Map<String, UnsafeCandidateRecord> unsafe = new HashMap<>();
        try {
            List<String> lines = Files.readAllLines(path, StandardCharsets.UTF_8);
            for (String line : lines) {
                if (line == null || line.isBlank() || line.startsWith("#")) {
                    continue;
                }
                String[] cols = line.split("\t", 5);
                if (cols.length < 5) {
                    continue;
                }
                if (!"UNSAFE".equals(cols[1])) {
                    continue;
                }
                if (!contextSignature.equals(cols[4])) {
                    continue;
                }
                unsafe.put(cols[0], new UnsafeCandidateRecord(cols[0], cols[2], cols[3], cols[4]));
            }
            return new UnsafeCandidateHistory(contextSignature, unsafe);
        } catch (IOException e) {
            return empty(contextSignature);
        }
    }

    public boolean isUnsafe(String fingerprint) {
        return unsafeByFingerprint.containsKey(fingerprint);
    }

    public void markUnsafe(String fingerprint, String candidateName, String reason) {
        if (unsafeByFingerprint.containsKey(fingerprint)) {
            return;
        }
        String now = OffsetDateTime.now().toString();
        String cleanReason = sanitize(reason + " candidate=" + candidateName);
        unsafeByFingerprint.put(
                fingerprint,
                new UnsafeCandidateRecord(fingerprint, cleanReason, now, contextSignature)
        );
        dirty = true;
    }

    public void save(Path path) {
        if (!dirty) {
            return;
        }
        try {
            if (path.getParent() != null) {
                Files.createDirectories(path.getParent());
            }
            List<String> lines = new ArrayList<>();
            lines.add("# fingerprint\tstatus\treason\ttimestamp\tcontext");
            for (UnsafeCandidateRecord record : unsafeByFingerprint.values()) {
                lines.add(record.toLine());
            }
            Files.write(path, lines, StandardCharsets.UTF_8);
            dirty = false;
        } catch (IOException e) {
            throw new IllegalStateException("Failed to write autotune candidate history", e);
        }
    }

    private static String sanitize(String value) {
        return value
                .replace('\t', ' ')
                .replace('\n', ' ')
                .replace('\r', ' ');
    }

    private record UnsafeCandidateRecord(
            String fingerprint,
            String reason,
            String timestamp,
            String context
    ) {
        private String toLine() {
            return fingerprint + "\tUNSAFE\t" + reason + "\t" + timestamp + "\t" + context;
        }
    }
}
