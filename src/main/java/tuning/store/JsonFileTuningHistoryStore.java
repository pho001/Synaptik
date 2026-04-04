package tuning.store;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

public final class JsonFileTuningHistoryStore implements TuningHistoryStore {
    @Override
    public void append(Path path, TuningHistoryEntry entry) {
        if (path == null) {
            throw new IllegalArgumentException("path cannot be null");
        }
        if (entry == null) {
            throw new IllegalArgumentException("entry cannot be null");
        }
        try {
            if (path.getParent() != null) {
                Files.createDirectories(path.getParent());
            }
            List<String> lines = Files.exists(path)
                    ? Files.readAllLines(path, StandardCharsets.UTF_8)
                    : new ArrayList<>();
            lines.add(toJson(entry));
            Files.write(path, lines, StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new IllegalStateException("Failed to append tuning history to " + path, e);
        }
    }

    @Override
    public List<TuningHistoryEntry> loadAll(Path path) {
        if (path == null || !Files.exists(path)) {
            return List.of();
        }
        try {
            List<String> lines = Files.readAllLines(path, StandardCharsets.UTF_8);
            List<TuningHistoryEntry> out = new ArrayList<>();
            for (String line : lines) {
                if (line == null || line.isBlank()) {
                    continue;
                }
                out.add(parse(line));
            }
            return List.copyOf(out);
        } catch (IOException e) {
            throw new IllegalStateException("Failed to read tuning history from " + path, e);
        }
    }

    private static String toJson(TuningHistoryEntry entry) {
        return "{"
                + "\"fingerprint\":\"" + escape(entry.fingerprint()) + "\","
                + "\"candidateName\":\"" + escape(entry.candidateName()) + "\","
                + "\"valid\":" + entry.valid() + ","
                + "\"medianMs\":" + String.format(Locale.US, "%.8f", entry.medianMs()) + ","
                + "\"meanMs\":" + String.format(Locale.US, "%.8f", entry.meanMs()) + ","
                + "\"score\":" + String.format(Locale.US, "%.8f", entry.score()) + ","
                + "\"failureReason\":\"" + escape(entry.failureReason()) + "\","
                + "\"summary\":\"" + escape(entry.summary()) + "\","
                + "\"timestamp\":\"" + entry.timestamp() + "\","
                + "\"hardwareKey\":\"" + escape(entry.hardware().key()) + "\","
                + "\"workloadKey\":\"" + escape(entry.workload().key()) + "\""
                + "}";
    }

    private static TuningHistoryEntry parse(String line) {
        return new TuningHistoryEntry(
                findString(line, "fingerprint", ""),
                findString(line, "candidateName", "candidate"),
                findBoolean(line, "valid", false),
                findDouble(line, "medianMs", Double.POSITIVE_INFINITY),
                findDouble(line, "meanMs", Double.POSITIVE_INFINITY),
                findDouble(line, "score", Double.POSITIVE_INFINITY),
                findString(line, "failureReason", ""),
                findString(line, "summary", ""),
                java.time.OffsetDateTime.parse(findString(line, "timestamp", java.time.OffsetDateTime.now().toString())),
                HardwareFingerprint.fromKey(findString(line, "hardwareKey", "")),
                WorkloadFingerprint.fromKey(findString(line, "workloadKey", ""))
        );
    }

    private static String escape(String value) {
        return value == null ? "" : value.replace("\\", "\\\\").replace("\"", "\\\"");
    }

    private static String findString(String json, String key, String defaultValue) {
        java.util.regex.Matcher matcher = java.util.regex.Pattern
                .compile("\"" + java.util.regex.Pattern.quote(key) + "\"\\s*:\\s*\"([^\"]*)\"")
                .matcher(json);
        return matcher.find() ? matcher.group(1) : defaultValue;
    }

    private static double findDouble(String json, String key, double defaultValue) {
        java.util.regex.Matcher matcher = java.util.regex.Pattern
                .compile("\"" + java.util.regex.Pattern.quote(key) + "\"\\s*:\\s*([-+0-9.eE]+)")
                .matcher(json);
        return matcher.find() ? Double.parseDouble(matcher.group(1)) : defaultValue;
    }

    private static boolean findBoolean(String json, String key, boolean defaultValue) {
        java.util.regex.Matcher matcher = java.util.regex.Pattern
                .compile("\"" + java.util.regex.Pattern.quote(key) + "\"\\s*:\\s*(true|false)")
                .matcher(json);
        return matcher.find() ? Boolean.parseBoolean(matcher.group(1)) : defaultValue;
    }
}
