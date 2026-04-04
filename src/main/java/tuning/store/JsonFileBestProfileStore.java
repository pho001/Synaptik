package tuning.store;

import config.profile.ExecutionProfile;
import config.profile.ExecutionProfileIO;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.OffsetDateTime;
import java.util.Locale;
import java.util.Optional;

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
            String profileBody = extractObject(json, "profile");
            if (profileBody == null) {
                return Optional.empty();
            }
            ExecutionProfile fallback = new ExecutionProfile(
                    "loaded-best",
                    "loaded-best",
                    tensor.DataType.FLOAT64,
                    backend.runtime.ExecutionMode.FORWARD,
                    config.optimizer.OptimizerConfig.inferenceDefaults(),
                    config.runtime.RuntimeConfig.inferenceDefaults()
            );
            ExecutionProfile profile = ExecutionProfileIO.fromJsonOrDefault(profileBody, fallback);
            return Optional.of(new BestProfileRecord(
                    HardwareFingerprint.fromKey(hardwareKey),
                    WorkloadFingerprint.fromKey(workloadKey),
                    profile,
                    score,
                    OffsetDateTime.parse(updatedAt)
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

    private static String escape(String value) {
        return value == null ? "" : value.replace("\\", "\\\\").replace("\"", "\\\"");
    }
}
