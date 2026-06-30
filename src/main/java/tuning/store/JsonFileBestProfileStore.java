package tuning.store;

import config.profile.ExecutionProfile;
import config.profile.ExecutionProfileIO;
import tuning.candidate.CandidateKind;
import tuning.candidate.CandidateMetadata;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.OffsetDateTime;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public final class JsonFileBestProfileStore implements BestProfileStore {
    @Override
    public void save(Path path, BestProfileRecord record) {
        if (path == null) {
            throw new IllegalArgumentException("path cannot be null");
        }
        if (record == null) {
            throw new IllegalArgumentException("record cannot be null");
        }
        String profileJson = ExecutionProfileIO.toJson(record.profile()).replace("\n", "\n    ");
        String json = "{\n"
                + "  \"score\": " + String.format(Locale.US, "%.8f", record.score()) + ",\n"
                + "  \"updatedAt\": \"" + record.updatedAt() + "\",\n"
                + "  \"hardwareKey\": \"" + escape(record.hardware().key()) + "\",\n"
                + "  \"workloadKey\": \"" + escape(record.workload().key()) + "\",\n"
                + "  \"autotuneKind\": \"" + escape(record.autotuneKind()) + "\",\n"
                + "  \"graphAutotuneMode\": \"" + escape(record.graphAutotuneMode()) + "\",\n"
                + "  \"candidateKind\": \"" + record.candidateKind().name() + "\",\n"
                + "  \"runtimeProfileId\": \"" + escape(record.runtimeProfileId()) + "\",\n"
                + "  \"productionEligible\": " + record.productionEligible() + ",\n"
                + "  \"candidateMetadata\": " + mapJson(record.candidateMetadata().toMap()) + ",\n"
                + "  \"profile\": " + profileJson + "\n"
                + "}\n";
        try {
            if (path.getParent() != null) {
                Files.createDirectories(path.getParent());
            }
            Files.writeString(path, json, StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new IllegalStateException("Failed to write best profile to " + path, e);
        }
    }

    @Override
    public Optional<BestProfileRecord> load(Path path) {
        if (path == null || !Files.exists(path)) {
            return Optional.empty();
        }
        try {
            String json = Files.readString(path, StandardCharsets.UTF_8);
            double score = findDouble(json, "score", Double.POSITIVE_INFINITY);
            String updatedAt = findString(json, "updatedAt", OffsetDateTime.now().toString());
            String hardwareKey = findString(json, "hardwareKey", "");
            String workloadKey = findString(json, "workloadKey", "");
            String autotuneKind = findString(json, "autotuneKind", "legacy");
            String graphAutotuneMode = findString(json, "graphAutotuneMode", "");
            CandidateKind candidateKind = findEnum(json, "candidateKind", CandidateKind.GENERIC, CandidateKind.class);
            String runtimeProfileId = findString(json, "runtimeProfileId", "");
            boolean productionEligible = findBoolean(json, "productionEligible", true);
            CandidateMetadata candidateMetadata = CandidateMetadata.fromMap(parseStringMap(extractObject(json, "candidateMetadata")));
            String profileBody = extractObject(json, "profile");
            if (profileBody == null) {
                return Optional.empty();
            }
            ExecutionProfile fallback = new ExecutionProfile(
                    "loaded-best",
                    "loaded-best",
                    tensor.DataType.FLOAT64,
                    runtime.contract.ExecutionMode.FORWARD,
                    config.compile.CompileConfig.inference(),
                    config.runtime.RuntimeConfig.inferenceDefaults()
            );
            ExecutionProfile profile = ExecutionProfileIO.fromJsonStrict(profileBody, fallback);
            return Optional.of(new BestProfileRecord(
                    HardwareFingerprint.fromKey(hardwareKey),
                    WorkloadFingerprint.fromKey(workloadKey),
                    profile,
                    score,
                    OffsetDateTime.parse(updatedAt),
                    autotuneKind,
                    graphAutotuneMode,
                    candidateKind,
                    candidateMetadata,
                    runtimeProfileId,
                    productionEligible
            ));
        } catch (Exception e) {
            return Optional.empty();
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
            if (c == '{') depth++;
            if (c == '}') {
                depth--;
                if (depth == 0) {
                    return json.substring(braceStart, i + 1);
                }
            }
        }
        return null;
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
                sb.append(", ");
            }
            first = false;
            sb.append("\"").append(escape(entry.getKey())).append("\": \"")
                    .append(escape(entry.getValue())).append("\"");
        }
        sb.append("}");
        return sb.toString();
    }

    private static String escape(String value) {
        return value == null ? "" : value.replace("\\", "\\\\").replace("\"", "\\\"");
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
