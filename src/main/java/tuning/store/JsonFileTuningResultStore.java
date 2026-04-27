package tuning.store;

import tuning.autotune.report.JsonTuningResultRenderer;
import tuning.autotune.TuningResult;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

public final class JsonFileTuningResultStore implements TuningResultStore {
    @Override
    public void save(Path path, TuningResult result) {
        if (path == null) {
            throw new IllegalArgumentException("path cannot be null");
        }
        if (result == null) {
            throw new IllegalArgumentException("result cannot be null");
        }
        try {
            if (path.getParent() != null) {
                Files.createDirectories(path.getParent());
            }
            Files.writeString(path, JsonTuningResultRenderer.render(result), StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new IllegalStateException("Failed to write tuning result to " + path, e);
        }
    }
}
