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
        return List.of();
    }

    private static String toJson(TuningHistoryEntry entry) {
        return "{"
                + "\"candidateName\":\"" + escape(entry.candidateName()) + "\","
                + "\"valid\":" + entry.valid() + ","
                + "\"medianMs\":" + String.format(Locale.US, "%.8f", entry.medianMs()) + ","
                + "\"meanMs\":" + String.format(Locale.US, "%.8f", entry.meanMs()) + ","
                + "\"score\":" + String.format(Locale.US, "%.8f", entry.score()) + ","
                + "\"summary\":\"" + escape(entry.summary()) + "\","
                + "\"timestamp\":\"" + entry.timestamp() + "\","
                + "\"hardwareKey\":\"" + escape(entry.hardware().key()) + "\","
                + "\"workloadKey\":\"" + escape(entry.workload().key()) + "\""
                + "}";
    }

    private static String escape(String value) {
        return value == null ? "" : value.replace("\\", "\\\\").replace("\"", "\\\"");
    }
}
