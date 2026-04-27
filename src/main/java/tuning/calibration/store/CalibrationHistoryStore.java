package tuning.calibration.store;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

public final class CalibrationHistoryStore {
    public void append(Path path, CalibrationRunRecord record) {
        if (path == null || record == null) {
            return;
        }
        try {
            if (path.getParent() != null) {
                Files.createDirectories(path.getParent());
            }
            Files.writeString(
                    path,
                    record.toJson() + System.lineSeparator(),
                    StandardCharsets.UTF_8,
                    Files.exists(path) ? java.nio.file.StandardOpenOption.APPEND : java.nio.file.StandardOpenOption.CREATE
            );
        } catch (IOException e) {
            throw new IllegalStateException("Failed to append calibration history to " + path, e);
        }
    }
}
