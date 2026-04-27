package tuning.calibration.store;

import config.profile.PlatformRuntimeProfile;
import config.profile.PlatformRuntimeProfileIO;
import tuning.calibration.report.TextPlatformCalibrationResultRenderer;
import tuning.calibration.PlatformCalibrationResult;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

public final class PlatformCalibrationSaveHelper {
    private PlatformCalibrationSaveHelper() {
    }

    public static void saveAll(
            PlatformCalibrationResult result,
            Path profilePath,
            Path jsonReportPath,
            Path textReportPath
    ) {
        if (result == null) {
            throw new IllegalArgumentException("result cannot be null");
        }
        saveProfile(result.finalRuntimeProfile(), profilePath);
        saveJson(result, jsonReportPath);
        saveText(result, textReportPath);
    }

    public static void saveProfile(PlatformRuntimeProfile profile, Path path) {
        if (profile == null || path == null) {
            return;
        }
        PlatformRuntimeProfileIO.save(path, profile);
    }

    public static void saveJson(PlatformCalibrationResult result, Path path) {
        if (result == null || path == null) {
            return;
        }
        new JsonFilePlatformCalibrationResultStore().save(path, result);
    }

    public static void saveText(PlatformCalibrationResult result, Path path) {
        if (result == null || path == null) {
            return;
        }
        try {
            if (path.getParent() != null) {
                Files.createDirectories(path.getParent());
            }
            Files.writeString(path, TextPlatformCalibrationResultRenderer.render(result), StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new IllegalStateException("Failed to write platform calibration text report to " + path, e);
        }
    }
}
