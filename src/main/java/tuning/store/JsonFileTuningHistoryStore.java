package tuning.store;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import tuning.candidate.CandidateKind;
import tuning.candidate.CandidateMetadata;

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
                + "\"workloadKey\":\"" + escape(entry.workload().key()) + "\","
                + "\"candidateKind\":\"" + entry.candidateKind().name() + "\","
                + "\"runtimeProfileId\":\"" + escape(entry.runtimeProfileId()) + "\","
                + "\"productionEligible\":" + entry.productionEligible() + ","
                + "\"candidateMetadata\":" + mapJson(entry.candidateMetadata().toMap())
                + "}";
    }

    private static TuningHistoryEntry parse(String line) {
        CandidateKind candidateKind = findEnum(line, "candidateKind", CandidateKind.GENERIC, CandidateKind.class);
        CandidateMetadata candidateMetadata = CandidateMetadata.fromMap(parseStringMap(extractObject(line, "candidateMetadata")));
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
                WorkloadFingerprint.fromKey(findString(line, "workloadKey", "")),
                candidateKind,
                candidateMetadata,
                findString(line, "runtimeProfileId", ""),
                findBoolean(line, "productionEligible", true)
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

    private static <E extends Enum<E>> E findEnum(String json, String key, E defaultValue, Class<E> enumClass) {
        String value = findString(json, key, "");
        if (value.isBlank()) {
            return defaultValue;
        }
        try {
            return Enum.valueOf(enumClass, value);
        } catch (IllegalArgumentException ignored) {
            return defaultValue;
        }
    }

    private static String extractObject(String json, String key) {
        int keyIndex = json.indexOf("\"" + key + "\"");
        if (keyIndex < 0) {
            return null;
        }
        int braceStart = json.indexOf('{', keyIndex);
        if (braceStart < 0) {
            return null;
        }
        int depth = 0;
        for (int i = braceStart; i < json.length(); i++) {
            char c = json.charAt(i);
            if (c == '{') {
                depth++;
            }
            if (c == '}') {
                depth--;
                if (depth == 0) {
                    return json.substring(braceStart, i + 1);
                }
            }
        }
        return null;
    }

    private static Map<String, String> parseStringMap(String json) {
        if (json == null || json.isBlank()) {
            return Map.of();
        }
        LinkedHashMap<String, String> out = new LinkedHashMap<>();
        Matcher matcher = Pattern.compile("\"((?:[^\"\\\\]|\\\\.)*)\"\\s*:\\s*\"((?:[^\"\\\\]|\\\\.)*)\"")
                .matcher(json);
        while (matcher.find()) {
            out.put(unescape(matcher.group(1)), unescape(matcher.group(2)));
        }
        return Map.copyOf(out);
    }

    private static String mapJson(Map<String, String> values) {
        if (values == null || values.isEmpty()) {
            return "{}";
        }
        StringBuilder sb = new StringBuilder("{");
        boolean first = true;
        for (Map.Entry<String, String> entry : values.entrySet()) {
            if (!first) {
                sb.append(",");
            }
            first = false;
            sb.append("\"").append(escape(entry.getKey())).append("\":\"")
                    .append(escape(entry.getValue())).append("\"");
        }
        sb.append("}");
        return sb.toString();
    }

    private static String unescape(String value) {
        if (value == null || value.isEmpty()) {
            return "";
        }
        StringBuilder out = new StringBuilder(value.length());
        boolean escaping = false;
        for (int i = 0; i < value.length(); i++) {
            char c = value.charAt(i);
            if (escaping) {
                out.append(c);
                escaping = false;
            } else if (c == '\\') {
                escaping = true;
            } else {
                out.append(c);
            }
        }
        if (escaping) {
            out.append('\\');
        }
        return out.toString();
    }
}
