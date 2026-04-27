package tuning.calibration.store;

import tuning.calibration.report.JsonPlatformCalibrationResultRenderer;
import tuning.calibration.PlatformCalibrationResult;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

public final class JsonFilePlatformCalibrationResultStore implements PlatformCalibrationResultStore {
    @Override
    public void save(Path path, PlatformCalibrationResult result) {
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
            Files.writeString(path, JsonPlatformCalibrationResultRenderer.render(result), StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new IllegalStateException("Failed to write platform calibration result to " + path, e);
        }
    }
}
