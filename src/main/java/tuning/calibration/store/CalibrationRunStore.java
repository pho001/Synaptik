package tuning.calibration.store;

import config.profile.PlatformRuntimeProfile;
import config.profile.PlatformRuntimeProfileIO;
import tuning.calibration.PlatformCalibrationResult;
import tuning.calibration.PlatformCalibrationStepResult;
import tuning.calibration.family.CalibrationFamilyRegistry;
import tuning.calibration.report.JsonPlatformCalibrationResultRenderer;
import tuning.calibration.report.TextPlatformCalibrationResultRenderer;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

public final class CalibrationRunStore {
    private final CalibrationArtifactLayout layout;
    private final CalibrationHistoryStore historyStore = new CalibrationHistoryStore();

    public CalibrationRunStore(CalibrationArtifactLayout layout) {
        this.layout = layout;
    }

    public void writeManifest(CalibrationRunManifest manifest) {
        writeString(layout.runManifestPath(manifest.runId()), manifest.toJson());
        writeString(layout.rootManifestPath(), manifest.toJson());
    }

    public void saveStep(
            String runId,
            String dtype,
            String mode,
            int passIndex,
            int passCount,
            PlatformCalibrationResult result,
            PlatformCalibrationStepResult step
    ) {
        Path resultJson = layout.resultJsonPath(runId, dtype, mode, step.family());
        Path resultText = layout.resultTextPath(runId, dtype, mode, step.family());
        Path selectedProfile = layout.selectedProfilePath(runId, dtype, mode, step.family());
        writeString(resultJson, JsonPlatformCalibrationResultRenderer.render(result));
        writeString(resultText, TextPlatformCalibrationResultRenderer.render(result));
        writeProfile(selectedProfile, step.selectedRuntimeProfile());
        writeCandidates(layout.candidatesPath(runId, dtype, mode, step.family()), step);
        historyStore.append(
                layout.historyPath(dtype, mode, step.family()),
                new CalibrationRunRecord(runId, dtype, mode, passIndex, passCount, step, selectedProfile)
        );
    }

    public void publishLatest(
            CalibrationRunManifest manifest,
            String dtype,
            String mode,
            PlatformRuntimeProfile profile
    ) {
        Path profilePath = layout.latestProfilePath(dtype, mode);
        writeProfileAtomic(profilePath, profile);
        writeString(layout.latestManifestPath(dtype, mode), manifest.latestJson(dtype, profilePath.toString()));
    }

    private static void writeCandidates(Path path, PlatformCalibrationStepResult step) {
        StringBuilder sb = new StringBuilder();
        for (var candidate : step.candidateSummaries()) {
            sb.append("{")
                    .append("\"family\":\"").append(CalibrationFamilyRegistry.spec(step.family()).cliName()).append("\",")
                    .append("\"candidateId\":\"").append(escape(candidate.candidateId())).append("\",")
                    .append("\"valid\":").append(candidate.score().valid()).append(",")
                    .append("\"score\":").append(Double.isFinite(candidate.score().score()) ? candidate.score().score() : "null")
                    .append("}")
                    .append(System.lineSeparator());
        }
        writeString(path, sb.toString());
    }

    private static void writeProfile(Path path, PlatformRuntimeProfile profile) {
        PlatformRuntimeProfileIO.save(path, profile);
    }

    private static void writeProfileAtomic(Path path, PlatformRuntimeProfile profile) {
        if (path == null || profile == null) {
            return;
        }
        try {
            if (path.getParent() != null) {
                Files.createDirectories(path.getParent());
            }
            Path temp = path.resolveSibling(path.getFileName() + ".tmp");
            Files.writeString(temp, PlatformRuntimeProfileIO.toJson(profile), StandardCharsets.UTF_8);
            try {
                Files.move(temp, path, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
            } catch (IOException atomicFailure) {
                Files.move(temp, path, StandardCopyOption.REPLACE_EXISTING);
            }
        } catch (IOException e) {
            throw new IllegalStateException("Failed to publish latest calibration profile to " + path, e);
        }
    }

    private static void writeString(Path path, String content) {
        if (path == null) {
            return;
        }
        try {
            if (path.getParent() != null) {
                Files.createDirectories(path.getParent());
            }
            Files.writeString(path, content == null ? "" : content, StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new IllegalStateException("Failed to write calibration artifact to " + path, e);
        }
    }

    private static String escape(String value) {
        return value == null ? "" : value.replace("\\", "\\\\").replace("\"", "\\\"");
    }
}
